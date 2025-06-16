import numpy as np

from data_reader_utils import (get_filepaths_dict, get_input_signal_position,
                               import_arousal, import_arousal_mat, import_mat)
from PhysionetDataset import PhysionetDataset
from visualization_utils import plot_signal_segment


class SleepNetDataset(PhysionetDataset):
    def __init__(self, split, settings, return_arousal_signals=False):
        super().__init__(split, settings, return_arousal_signals)
        self.reduction_factor = 50
        self.sample_data_limit = 7*3600*50

    def pad(self, signals):
        for n in range(len(signals)):
            if signals.shape[0] < self.sample_data_limit:
                # Zero Pad
                needed_length = self.sample_data_limit - signals.shape[0]
                extension = np.zeros(shape=(needed_length, signals.shape[1]))
                extension[::, -3::] = -1.0
                signals = np.concatenate([signals, extension], axis=0)

            elif signals.shape[0] > self.sample_data_limit:
                # Chop
                signals = signals[0:self.sample_data_limit, ::]

        return signals

    def pad1(self, signals):
        if len(signals) < self.sample_data_limit:
            # Zero Pad
            needed_length = self.sample_data_limit - len(signals)
            extension = -np.ones(shape=(needed_length))
            signals = np.concatenate([signals, extension], axis=0)

        elif len(signals) > self.sample_data_limit:
            # Chop
            signals = signals[0:self.sample_data_limit]

        return signals

    def preprocess_input_signals(self, input_signals):
        from scipy.signal import fftconvolve

        input_signals = input_signals.astype(np.float64)
        keep_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # remove ecg
        filt_coeff = np.array([0.00637849379422531, 0.00543091599801427, -0.00255136650039784, -0.0123109503066702,
                          -0.0137267267561505, -0.000943230632358082, 0.0191919895027550, 0.0287148886882440,
                          0.0123598773891149, -0.0256928886371578, -0.0570987715759348, -0.0446385294777459,
                          0.0303553522906817, 0.148402006671856, 0.257171285176269, 0.301282456398562,
                          0.257171285176269, 0.148402006671856, 0.0303553522906817, -0.0446385294777459,
                          -0.0570987715759348, -0.0256928886371578, 0.0123598773891149, 0.0287148886882440,
                          0.0191919895027550, -0.000943230632358082, -0.0137267267561505, -0.0123109503066702,
                          -0.00255136650039784, 0.00543091599801427, 0.00637849379422531])

        for n in range(input_signals.shape[1]):
            input_signals[::, n] = np.convolve(input_signals[::, n], filt_coeff, mode='same')
        input_signals = input_signals[0::4, keep_channels]

        input_signals[::, 11] += -32768.0
        input_signals[::, 11] /= 65535.0
        input_signals[::, 11] -= 0.5

        kernel_size = (50*18*60)+1

        center = np.mean(input_signals, axis=0)
        scale = np.std(input_signals, axis=0)
        scale[scale == 0] = 1.0
        input_signals = (input_signals - center) / scale

        center = np.zeros(input_signals.shape)
        for n in range(input_signals.shape[1]):
            center[::, n] = fftconvolve(input_signals[::, n], np.ones(shape=(kernel_size,))/kernel_size, mode='same')

        # Exclude SAO2
        center[::, 11] = 0.0
        center[np.isnan(center) | np.isinf(center)] = 0.0
        input_signals = input_signals - center

        # Compute and remove the rms with FFT convolution of squared signal
        scale = np.ones(input_signals.shape)
        for n in range(input_signals.shape[1]):
            temp = fftconvolve(np.square(input_signals[::, n]), np.ones(shape=(kernel_size,))/kernel_size, mode='same')

            # Deal with negative values (mathematically, it should never be negative, but fft artifacts can cause this)
            temp[temp < 0] = 0.0

            # Deal with invalid values
            invalidIndices = np.isnan(temp) | np.isinf(temp)
            temp[invalidIndices] = 0.0
            maxTemp = np.max(temp)
            temp[invalidIndices] = maxTemp

            # Finish rms calculation
            scale[::, n] = np.sqrt(temp)

        # Exclude SAO2
        scale[::, 11] = 1.0

        scale[(scale == 0) | np.isinf(scale) | np.isnan(scale)] = 1.0  # To correct for record 12 that has a zero amplitude chest signal
        input_signals = input_signals / scale

        # Enforce dataLimitInHours hour length with chopping / zero padding for memory usage stability and effficiency in cuDNN
        if self.split == 'train':
            input_signals = self.pad(input_signals)

        input_signals = input_signals.astype(np.float32)

        return input_signals

    def combine_outputs(self, output_signal, arousal_signals):
        arousal_annotations = output_signal
        apnea_hypopnea_annotations = np.zeros(shape=(len(output_signal)))
        sleep_stage_annotations = np.full((len(output_signal)), -1.)

        apnea_hypopnea_annotations = (arousal_signals['obst'].astype(bool) | arousal_signals['cent'].astype(bool) | arousal_signals['mix'].astype(bool)).astype(float)
        sleep_stage_annotations[arousal_signals['wake'].astype(bool)] = 0.
        sleep_stage_annotations[arousal_signals['rem'].astype(bool)
                                | arousal_signals['nrem1'].astype(bool)
                                | arousal_signals['nrem2'].astype(bool)
                                | arousal_signals['nrem3'].astype(bool)] = 1.

        sleep_stage_annotations[(arousal_annotations < -0.5).astype(bool) & ~(apnea_hypopnea_annotations.astype(bool)) & sleep_stage_annotations.astype(bool)] = 0.
        sleep_stage_annotations[(arousal_annotations < -0.5).astype(bool) & apnea_hypopnea_annotations.astype(bool) & ~(sleep_stage_annotations.astype(bool))] = 1.

        if self.split == 'train':
            # 200 Hz -> 50 Hz
            arousal_annotations = arousal_annotations[0::4]
            apnea_hypopnea_annotations = apnea_hypopnea_annotations[0::4]
            sleep_stage_annotations = sleep_stage_annotations[0::4]

            # pad
            arousal_annotations = self.pad1(arousal_annotations)
            apnea_hypopnea_annotations = self.pad1(apnea_hypopnea_annotations)
            sleep_stage_annotations = self.pad1(sleep_stage_annotations)

            arousal_annotations = arousal_annotations[0::self.reduction_factor]
            apnea_hypopnea_annotations = apnea_hypopnea_annotations[0::self.reduction_factor]
            sleep_stage_annotations = sleep_stage_annotations[0::self.reduction_factor]

        # bool -> long
        arousal_annotations = arousal_annotations.astype(np.int32).reshape(-1)
        apnea_hypopnea_annotations = apnea_hypopnea_annotations.astype(np.int32).reshape(-1)
        sleep_stage_annotations = sleep_stage_annotations.astype(np.int32).reshape(-1)

        output_signals = (arousal_annotations, apnea_hypopnea_annotations, sleep_stage_annotations)

        return output_signals


def main():
    import sys

    import torch
    import yaml
    from torch.utils.data import DataLoader

    if len(sys.argv) == 2:
        settings_name = sys.argv[1].strip()
    else:
        print('How to use: python dataset.py settings_local.yaml')
        return

    torch.manual_seed(0)

    with open(settings_name, 'r') as file:
        settings = yaml.safe_load(file)

    train_data = SleepNetDataset('train', settings, True)
    batch_size = 1  # settings['train_batch_size']
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)


    for batch_idx, (input_signals, output_signals, arousal_signals) in enumerate(train_dataloader):
        input_signals = input_signals.squeeze(0)
        try:
            output_signals = output_signals.squeeze(0)
        except:
            pass
        arousal_signals = {k: v.squeeze(axis=0) for k, v in arousal_signals.items()}  # squeeze dict

        print('batch_idx: ' + str(batch_idx))
        print('input shape: ' + str(input_signals.shape))
        try:
            print('output shape: ' + str(output_signals.shape))
        except:
            print('output shape: ' + str(output_signals[0].shape))

        starting = 0  # 100_000
        ending = input_signals.shape[0] # 4_000_000

        # output_signals = output_signals[0]
        # print(output_signals)
        # print(output_signals.shape)
        # plot_signal_segment(input_signals, output_signals, arousal_signals, None, starting, ending)
        # plot_signal_segment(input_signals, output_signals, None, None, starting, ending)
        plot_signal_segment(input_signals, None, None, None, starting, ending)

        exit()


if __name__ == '__main__':
    main()
