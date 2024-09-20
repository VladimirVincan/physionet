import os

import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader, Dataset

import physionetchallenge2018_lib as phyc
from PhysionetDataset import PhysionetDataset


def main():
    config_file = 'mamba/config_fmle.yaml'
    config_file = 'mamba/config_local.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    listdir = os.listdir(config['train_dataset'])

    max_accum = None
    min_accum = None
    sum_accum = None
    count_accum = None

    for idx, folder_name in enumerate(listdir):
        record_name = os.path.join(config['train_dataset'], folder_name, folder_name)

        header_file = record_name + '.hea'
        signal_file = record_name + '.mat'
        arousal_file = record_name + '-arousal.mat'

        signal_names, Fs, n_samples = phyc.import_signal_names(header_file)
        signal_names = list(np.append(signal_names, 'arousals'))

        # Convert this subject's data into a pandas dataframe
        this_data = phyc.get_subject_data(arousal_file, signal_file, signal_names)

        input_signals = this_data.loc[:, this_data.columns!='arousals']
        arousals = this_data.get(['arousals']).values

        if max_accum is None:
            max_accum = input_signals.max()
            min_accum = input_signals.min()
            sum_accum = input_signals.sum()
            count_accum = input_signals.count()
        else:
            max_accum = pd.concat([max_accum, input_signals.max()], axis=1).max(axis=1)
            min_accum = pd.concat([min_accum, input_signals.min()], axis=1).min(axis=1)
            sum_accum += input_signals.sum()
            count_accum += input_signals.count()

        # print(input_signals.shape)
        # print(input_signals.max())
        # print(input_signals.min())
        # print(input_signals.sum())
        # print(input_signals.count())
        # exit()

    print(max_accum)
    print(min_accum)
    print(sum_accum)
    print(count_accum)


if __name__ == '__main__':
    main()
