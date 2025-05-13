import numpy as np
import torch
from torch.utils.data import Dataset

from data_reader_utils import (get_filepaths_dict, get_input_signal_position,
                               import_arousal, import_arousal_mat, import_mat)


class PhysionetDataset(Dataset):
    def __init__(self, split, settings):
        self.split = split
        self.settings = settings

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

        return input_signals, output_signals

    def preprocess_input_signals(self, input_signals):
        input_signals = input_signals.astype(np.float32)
        input_signals = torch.Tensor(input_signals)
        return input_signals

    def combine_outputs(self, output_signal, arousal_signals):
        output_signals = arousal_signals['rera']
        output_signals = torch.Tensor(output_signals)
        return output_signal


def main():
    import sys

    import torch
    import yaml
    from torch.utils.data import DataLoader

    if len(sys.argv) == 2:
        settings_name = sys.argv[1].strip()
    else:
        print('How to use: python main.py settings_local.yaml')
        return

    torch.manual_seed(0)

    with open(settings_name, 'r') as file:
        settings = yaml.safe_load(file)

    train_data = PhysionetDataset('train', settings)
    batch_size = 1  # settings['train_batch_size']
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    for batch_idx, (input_signals, output_signals) in enumerate(train_dataloader):
        print('batch_idx: ' + str(batch_idx))
        print('input shape: ' + str(input_signals.shape))
        print('output shape: ' + str(output_signals.shape))


if __name__ == '__main__':
    main()
