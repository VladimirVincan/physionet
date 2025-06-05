import numpy as np
import torch
from torch.utils.data import Dataset

from data_reader_utils import (get_filepaths_dict, get_input_signal_position,
                               import_arousal, import_arousal_mat, import_mat)
from PhysionetDataset import PhysionetDataset
from visualization_utils import plot_signal_segment


class DeepSleepDataset(PhysionetDataset):
    def __init__(self, split, settings, return_arousal_signals=False):
        super().__init__(split, settings, return_arousal_signals)
        self.ref555 = None

    def randomize_magnitude(self, input_signals):
        if self.split == 'train':
            factor = np.random.uniform(0.8, 1.25)  # alternative: 0.90 and 1.15
        else:
            return input_signals
        return input_signals * factor

    def randomize_padding(self, input_signals):
        total_length = 8_388_608  # 2**23
        signal_length = input_signals.shape[0]
        pad_length = total_length - signal_length

        if self.split == 'train' and self.settings['randomize_padding']:
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

    train_data = DeepSleepDataset('train', settings, True)
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
