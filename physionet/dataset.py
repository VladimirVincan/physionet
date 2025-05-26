import numpy as np
import torch
from torch.utils.data import Dataset

from data_reader_utils import (get_filepaths_dict, get_input_signal_position,
                               import_arousal, import_arousal_mat, import_mat)
from visualization_utils import plot_signal_segment


class PhysionetDataset(Dataset):
    def __init__(self, split, settings, return_arousal_signals=False):
        self.split = split
        self.settings = settings
        self.return_arousal_signals = return_arousal_signals
        self.init()

    def init(self):
        pass

    def __len__(self):
        return len(self.settings[self.split])

    def __getitem__(self, idx):
        filepaths_dict = get_filepaths_dict(self.settings, self.split, idx)
        input_signals = import_mat(filepaths_dict['mat_path'])
        output_signal, _ = import_arousal_mat(filepaths_dict['arousal_mat_path'])
        num_samples = output_signal.shape[0]  # (N, 1) -> N
        arousal_signals = import_arousal(filepaths_dict['wfdb_path'], num_samples)

        input_signals = self.preprocess_input_signals(input_signals)
        output_signals = self.combine_outputs(output_signal, arousal_signals)

        if self.return_arousal_signals:
            arousal_signals = self.preprocess_arousal_signals(arousal_signals)
            return input_signals, output_signals, arousal_signals
        elif self.split == 'test' or self.split == 'validation':
            return input_signals, output_signals, filepaths_dict, num_samples
        # elif self.split == 'validation':
        #     return input_signals, output_signals, num_samples
        return input_signals, output_signals

    def preprocess_input_signals(self, input_signals):
        input_signals = input_signals.astype(np.float32)
        # input_signals = torch.Tensor(input_signals)
        return input_signals

    def preprocess_arousal_signals(self, arousal_signals):
        return arousal_signals

    def combine_outputs(self, output_signal, arousal_signals):
        # output_signals = arousal_signals['rera']
        # output_signals = torch.Tensor(output_signals)
        output_signals = output_signal
        return output_signals


class NormalizedPhysionetDataset(PhysionetDataset):
    def normalize_columns(self, signal, min_vals, max_vals):
        """
        Normalize a 2D array with custom min/max values for each column.

        Args:
            signal: np.ndarray of shape (T, C) — T timesteps, C channels.
            min_vals: list or np.ndarray of shape (C,) — min per column.
            max_vals: list or np.ndarray of shape (C,) — max per column.

        Returns:
            normalized: np.ndarray of shape (T, C), values in [0, 1].
    """
        signal = signal.astype(np.float32)
        min_vals = np.asarray(min_vals).reshape(1, -1)
        max_vals = np.asarray(max_vals).reshape(1, -1)

        normalized = (signal - min_vals) / (max_vals - min_vals)
        return normalized

    def preprocess_input_signals(self, input_signals):

        # Apply min-max normalization per channel
        # ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
        min_vals = [-400, -400, -400, -400, -400, -400, -400, -400, -1000, -400, -400, 25000, -1800]
        max_vals = [ 400,  400,  400,  400,  400,  400,  400,  400,  1000,  400,  400, 35000,  1800]
        input_signals = np.clip(input_signals, min_vals, max_vals)
        # input_signals = self.normalize_columns(input_signals, min_vals, max_vals)

        input_signals = input_signals.astype(np.float32)
        return input_signals
        # return torch.tensor(input_signals)

class DeepSleepDataset(PhysionetDataset):
    def init(self):
        self.ref555 = None

    def clip(self, input_signals):
        min_vals = [-200, -200, -200, -200, -200, -200, -200, -200, -500, -400, -200, 25000, -1200]
        max_vals = [ 200,  200,  200,  200,  200,  200,  200,  200,  500,  400,  200, 35000,  1200]
        input_signals = np.clip(input_signals, min_vals, max_vals)
        min_vals = np.array(min_vals).reshape(-1, 1)  # Shape (13, 1)
        max_vals = np.array(max_vals).reshape(-1, 1)  # Shape (13, 1)
        denom = max_vals - min_vals
        input_signals = (input_signals - (max_vals.T + min_vals.T)/2) / denom.T
        return input_signals

    def randomize_magnitude(self, input_signals):
        if self.split == 'train':
            factor = np.random.uniform(0.8, 1.2)
        else:
            return input_signals
        return input_signals * factor

    def randomize_padding(self, input_signals):
        total_length = 8_388_608  # 2**23
        signal_length = input_signals.shape[0]
        pad_length = total_length - signal_length

        if self.split == 'train':
            left_pad = np.random.randint(0, pad_length + 1)
        else:
            left_pad = pad_length // 2
        right_pad = pad_length - left_pad
        input_signals = np.pad(input_signals, ((left_pad, right_pad), (0, 0)), mode='constant')
        self.left_pad = left_pad
        return input_signals

    def gaussian_normalization(self, input_signals):
        mean = np.mean(input_signals, axis=0)
        std = np.std(input_signals, axis=0)

        # Avoid division by zero
        std[std == 0] = 1.0

        input_signals = (input_signals - mean) / std
        return input_signals

    def quantile_normalization(self, input_signals):
        if self.ref555 is None:
            self.ref555=np.load(self.settings['ref555_file_path']).T

        d0 = input_signals.shape[1]
        s1 = float(self.ref555.shape[0]) # size in
        s2 = float(input_signals.shape[0]) # size out
        input_signals_new = input_signals.copy()

        for i in range(d0):
            tmp = np.interp(np.arange(s2)/(s2-1)*(s1-1), np.arange(s1), self.ref555[:, i])
            input_signals_new[np.argsort(input_signals[:, i]), i] = tmp
        return input_signals_new

    def preprocess_input_signals(self, input_signals):

        # input_signals = self.clip(input_signals)
        self.quantile_normalization(input_signals)
        # input_signals = self.gaussian_normalization(input_signals)
        input_signals = self.randomize_magnitude(input_signals)
        input_signals = self.randomize_padding(input_signals)
        input_signals = input_signals.astype(np.float32)

        return input_signals

    def preprocess_arousal_signals(self, arousal_signals):
        total_length = 8_388_608  # 2**23

        for key, lst in arousal_signals.items():
            signal_length = len(lst)
            pad_length = total_length - signal_length
            left_pad = pad_length // 2
            right_pad = pad_length - left_pad
            arousal_signals[key] = np.pad(lst, ((left_pad, right_pad)), mode='constant')
        return arousal_signals

    def smooth_labels(self, output_signals):
        smoothing = float(self.settings['smoothing'])
        return output_signals * (1.0 - smoothing) + (1 - output_signals) * smoothing

    def combine_outputs(self, output_signal, arousal_signals):
        # output_signal = arousal_signals['rera']
        total_length = 8_388_608  # 2**23
        signal_length = output_signal.shape[0]

        pad_length = total_length - signal_length
        # left_pad = pad_length // 2
        left_pad = self.left_pad
        right_pad = pad_length - left_pad

        output_signals = output_signal
        if self.split == 'train':
            output_signals = self.smooth_labels(output_signals)
        output_signals = np.pad(output_signals, ((left_pad, right_pad)), mode='constant', constant_values=-1)
        output_signals = output_signals.astype(np.float32)

        return output_signals


class SleepNetDataset(PhysionetDataset):
    def init(self):
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

        # correction according to Fig. 3:
        sleep_stage_annotations[arousal_signals['rem'].astype(bool)
                                | arousal_signals['nrem1'].astype(bool)
                                | arousal_signals['nrem2'].astype(bool)
                                | arousal_signals['nrem3'].astype(bool)] = 1.


        sleep_stage_annotations[(arousal_annotations < -0.5).astype(bool) & ~(apnea_hypopnea_annotations.astype(bool)) & sleep_stage_annotations.astype(bool)] = 0.
        sleep_stage_annotations[(arousal_annotations < -0.5).astype(bool) & apnea_hypopnea_annotations.astype(bool) & ~(sleep_stage_annotations.astype(bool))] = 1.

        # bool -> np.float32
        arousal_annotations = arousal_annotations.astype(np.float32)
        apnea_hypopnea_annotations = apnea_hypopnea_annotations.astype(np.float32)
        sleep_stage_annotations = sleep_stage_annotations.astype(np.float32)

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

    # train_data = PhysionetDataset('train', settings, True)
    # train_data = DeepSleepDataset('train', settings, True)
    train_data = SleepNetDataset('train', settings, True)
    # train_data = NormalizedPhysionetDataset('train', settings, True)
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

        # plot_signal_segment(input_signals, output_signals, arousal_signals, None, starting, ending)
        plot_signal_segment(input_signals, output_signals, None, None, starting, ending)

        exit()


if __name__ == '__main__':
    main()
