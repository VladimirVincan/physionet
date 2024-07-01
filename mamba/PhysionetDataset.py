import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import physionetchallenge2018_lib as phyc


class PhysionetDataset(Dataset):
    def __init__(self, dir='/mnt/lun1/physionet/challenge-2018/training'):
        self.dir = dir
        self.listdir = os.listdir(dir)

    def __len__(self):
        return 1
        return 994  # train
        return 989  # test

    def __getitem__(self, idx):
        record_name = os.path.join(self.dir, self.listdir[idx], self.listdir[idx])

        header_file = record_name + '.hea'
        signal_file = record_name + '.mat'
        arousal_file = record_name + '-arousal.mat'

        # Get the signal names from the header file
        signal_names, Fs, n_samples = phyc.import_signal_names(header_file)
        signal_names = list(np.append(signal_names, 'arousals'))

        # Convert this subject's data into a pandas dataframe
        this_data = phyc.get_subject_data(arousal_file, signal_file, signal_names)

        input_signals = this_data.loc[:, this_data.columns!='arousals']
        arousals = this_data.get(['arousals']).values

        input_signals = torch.Tensor(input_signals.values)
        arousals = torch.Tensor(arousals)

        return input_signals, arousals
