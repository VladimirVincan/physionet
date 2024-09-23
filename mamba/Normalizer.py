
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
    sqr_mean_accum = None

    data_sum = {
        'F3-M2':       [-7.321140e+05],
        'F4-M1':       [ 6.844050e+05],
        'C3-M2':       [-2.550050e+05],
        'C4-M1':       [ 8.429400e+04],
        'O1-M2':       [-1.422110e+05],
        'O2-M1':       [-1.779290e+05],
        'E1-M2':       [-2.189450e+05],
        'Chin1-Chin2': [-9.489200e+04],
        'ABD':         [ 1.685725e+07],
        'CHEST':       [-5.703330e+08],
        'AIRFLOW':     [ 3.955237e+06],
        'SaO2':        [ 1.218996e+14],
        'ECG':         [-1.716462e+06]
    }

    data_count = {
        'F3-M2':       [4396088400],
        'F4-M1':       [4396088400],
        'C3-M2':       [4396088400],
        'C4-M1':       [4396088400],
        'O1-M2':       [4396088400],
        'O2-M1':       [4396088400],
        'E1-M2':       [4396088400],
        'Chin1-Chin2': [4396088400],
        'ABD':         [4396088400],
        'CHEST':       [4396088400],
        'AIRFLOW':     [4396088400],
        'SaO2':        [4396088400],
        'ECG':         [4396088400]
    }

    df_sum = pd.DataFrame(data_sum)
    df_count = pd.DataFrame(data_count)

    df_mean = df_sum / df_count

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
            # print(input_signals)
            # print(input_signals - df_mean.iloc[0])
            # print((input_signals - df_mean.iloc[0])**2)
            sqr_mean_accum = ((input_signals - df_mean.iloc[0])**2).sum()
            # print(sqr_mean_accum)
            # max_accum = input_signals.max()
            # min_accum = input_signals.min()
            # sum_accum = input_signals.sum()
            # count_accum = input_signals.count()
        else:
            sqr_mean_accum += ((input_signals - df_mean.iloc[0])**2).sum()
            # max_accum = pd.concat([max_accum, input_signals.max()], axis=1).max(axis=1)
            # min_accum = pd.concat([min_accum, input_signals.min()], axis=1).min(axis=1)
            # sum_accum += input_signals.sum()
            # count_accum += input_signals.count()

        # print(input_signals.shape)
        # print(input_signals.max())
        # print(input_signals.min())
        # print(input_signals.sum())
        # print(input_signals.count())
        # exit()

    # print(sqr_mean_accum)
    sqr_mean_accum = sqr_mean_accum / df_count.iloc[0]
    # print(sqr_mean_accum)
    sqr_mean_accum = sqr_mean_accum.apply(np.sqrt)
    print(sqr_mean_accum)
    # print(max_accum)
    # print(min_accum)
    # print(sum_accum)
    # print(count_accum)


if __name__ == '__main__':
    main()
