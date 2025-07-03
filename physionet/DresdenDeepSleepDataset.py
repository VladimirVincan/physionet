import numpy as np
from omegaconf import OmegaConf

from data_reader_utils import (get_filepaths_dict, get_input_signal_position,
                               import_arousal, import_arousal_mat, import_mat)
from PhysionetDataset import PhysionetDataset
from visualization_utils import plot_signal_segment


class DresdenDeepSleepDataset(PhysionetDataset):
    def __init__(self, split, settings, return_arousal_signals=False):
        super().__init__(split, settings, return_arousal_signals)
        self.reduction_factor = 50
        self.sample_data_limit = 7*3600*50
        self.rng = np.random.default_rng(seed=0)

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

    def randomize_magnitude(self, input_signals):
        if self.split == 'train':
            factor = np.random.uniform(0.8, 1.25)  # alternative: 0.90 and 1.15
        else:
            return input_signals
        return input_signals * factor

    def randomize_padding(self, input_signals):
        total_length = 2_097_152  # 2**21
        signal_length = input_signals.shape[0]
        pad_length = total_length - signal_length

        if self.split == 'train' and self.settings['randomize_padding']:
            left_pad = np.random.randint(0, pad_length + 1)
        else:
            left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        input_signals = np.pad(input_signals, ((left_pad, right_pad), (0, 0)), mode='constant')
        self.left_pad = left_pad
        return input_signals, left_pad, right_pad

    def preprocess_input_signals(self, input_signals):
        """
        channel num:
        0) F3-M2
        1) F4-M1
        2) C3-M2
        3) C4-M1
        4) O1-M2
        5) O2-M1
        6) E1-M2
        7) Chin1-Chin2
        8) ABD
        9) CHEST
        10) AIRFLOW
        11) SaO2
        12) ECG

        eeg signals: 0 to 5
        eog signals: 6
        chin: 7

        used signals in the three channel input:
        1) eeg - detect frequency shifts
        2) eog - indicator of R sleep
        3) chin EMG - requirement of increased submental EMG in R sleep
        """
        from scipy.signal import fftconvolve

        input_signals = input_signals.astype(np.float64)
        keep_eeg = np.random.randint(6)
        keep_channels = [keep_eeg, 6, 7]
        # keep_channels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # remove ecg
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

        # input_signals[::, 11] += -32768.0
        # input_signals[::, 11] /= 65535.0
        # input_signals[::, 11] -= 0.5

        kernel_size = (50*18*60)+1

        center = np.mean(input_signals, axis=0)
        scale = np.std(input_signals, axis=0)
        scale[scale == 0] = 1.0
        input_signals = (input_signals - center) / scale

        center = np.zeros(input_signals.shape)
        for n in range(input_signals.shape[1]):
            center[::, n] = fftconvolve(input_signals[::, n], np.ones(shape=(kernel_size,))/kernel_size, mode='same')

        # Exclude SAO2
        # center[::, 11] = 0.0
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
        # scale[::, 11] = 1.0

        scale[(scale == 0) | np.isinf(scale) | np.isnan(scale)] = 1.0  # To correct for record 12 that has a zero amplitude chest signal
        input_signals = input_signals / scale

        # Enforce dataLimitInHours hour length with chopping / zero padding for memory usage stability and effficiency in cuDNN

        input_signals = self.randomize_magnitude(input_signals)
        input_signals, left_pad, right_pad = self.randomize_padding(input_signals)
        input_signals = input_signals.astype(np.float32)

        return input_signals, left_pad, right_pad

    def smooth_labels(self, output_signals):
        smoothing = float(self.settings['smoothing'])
        return output_signals * (1.0 - smoothing) + (1 - output_signals) * smoothing

    def combine_outputs(self, output_signal, arousal_signals):
        # output_signal = arousal_signals['rera']
        # print(self.split)
        if self.split == 'train':  # do not do padding in case of validation or test
            total_length = 2_097_152  # 2**21
            output_signal = output_signal[0::4]
            signal_length = output_signal.shape[0]

            pad_length = total_length - signal_length
            # left_pad = pad_length // 2
            left_pad = self.left_pad
            right_pad = pad_length - left_pad

            output_signal = self.smooth_labels(output_signal)
            output_signal = np.pad(output_signal, ((left_pad, right_pad)), mode='constant', constant_values=-1)
        output_signals = output_signal
        output_signals = output_signals.astype(np.float32)

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

    settings = OmegaConf.load(settings_name)
    splits = OmegaConf.load(OmegaConf.to_container(settings, resolve=True)['splits_yaml'])
    model = OmegaConf.load(OmegaConf.to_container(settings, resolve=True)['model_yaml'])
    settings = OmegaConf.merge(settings, splits, model)
    settings = OmegaConf.to_container(settings, resolve=True)

    train_data = DresdenDeepSleepDataset(settings['split'], settings, True)
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

        # TO VISUALIZE ONLY OUTPUT SIGNAL
        # import matplotlib.pyplot as plt
        # plt.plot(output_signals.numpy())
        # plt.title("1D Tensor Plot")
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.grid(True)
        # plt.show()

        exit()


if __name__ == '__main__':
    main()
