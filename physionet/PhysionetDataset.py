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

    def __len__(self):
        return len(self.settings[self.split])

    def __getitem__(self, idx):
        filepaths_dict = get_filepaths_dict(self.settings, self.split, idx)
        input_signals = import_mat(filepaths_dict['mat_path'])
        output_signal, _ = import_arousal_mat(filepaths_dict['arousal_mat_path'])
        num_samples = output_signal.shape[0]  # (N, 1) -> N
        arousal_signals = import_arousal(filepaths_dict['wfdb_path'], num_samples)

        input_signals, left_pad, right_pad = self.preprocess_input_signals(input_signals)
        output_signals = self.combine_outputs(output_signal, arousal_signals)

        if self.return_arousal_signals:
            arousal_signals = self.preprocess_arousal_signals(arousal_signals)
            return input_signals, output_signals, arousal_signals
        elif self.split == 'test' or self.split == 'validation':
            metadata = {
                'left_pad': left_pad,
                'right_pad': right_pad,
                'filepaths_dict': filepaths_dict,
                'num_samples': num_samples
            }
            return input_signals, output_signals, metadata
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
