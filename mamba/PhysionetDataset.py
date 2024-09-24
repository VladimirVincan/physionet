import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import physionetchallenge2018_lib as phyc


def normalize_dataset(input_signals):
    data_max = {
        'F3-M2':       [7898.0],
        'F4-M1':       [7752.0],
        'C3-M2':       [7892.0],
        'C4-M1':       [7895.0],
        'O1-M2':       [7522.0],
        'O2-M1':       [7457.0],
        'E1-M2':       [7654.0],
        'Chin1-Chin2': [6134.0],
        'ABD':         [8510.0],
        'CHEST':       [8492.0],
        'AIRFLOW':     [8394.0],
        'SaO2':        [32443.0],
        'ECG':         [5022.0]
    }

    data_min = {
        'F3-M2':       [-8440.0],
        'F4-M1':       [-8229.0],
        'C3-M2':       [-7568.0],
        'C4-M1':       [-7825.0],
        'O1-M2':       [-7062.0],
        'O2-M1':       [-7400.0],
        'E1-M2':       [-7747.0],
        'Chin1-Chin2': [-5424.0],
        'ABD':         [-8538.0],
        'CHEST':       [-8432.0],
        'AIRFLOW':     [-8176.0],
        'SaO2':        [-32443.0],
        'ECG':         [-4984.0]
    }

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

    data_stddev = {
        'F3-M2':       [0.874849],
        'F4-M1':       [0.990004],
        'C3-M2':       [0.736432],
        'C4-M1':       [0.881619],
        'O1-M2':       [0.833106],
        'O2-M1':       [0.927212],
        'E1-M2':       [1.124177],
        'Chin1-Chin2': [0.588864],
        'ABD':         [7.802699],
        'CHEST':       [6.833236],
        'AIRFLOW':     [0.529896],
        'SaO2':        [131.715823],
        'ECG':         [2.387984]
    }

    df_max = pd.DataFrame(data_max)
    df_min = pd.DataFrame(data_min)
    df_sum = pd.DataFrame(data_sum)
    df_count = pd.DataFrame(data_count)
    df_stddev = pd.DataFrame(data_stddev)

    df_mean = df_sum / df_count
    df_diff = df_max - df_min

    # print(input_signals)
    input_signals = (input_signals - df_mean.iloc[0]) / df_stddev.iloc[0]
    # print(input_signals)

    return input_signals


class PhysionetDataset(Dataset):
    def __init__(self, dir='/mnt/lun1/physionet/challenge-2018/training'):
        self.dir = dir
        self.listdir = os.listdir(dir)

    def __len__(self):
        return len(self.listdir)
        # return 994  # train
        # return 989  # test

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

        # normalize
        input_signals = normalize_dataset(input_signals)

        input_signals = torch.Tensor(input_signals.values)
        arousals = torch.Tensor(arousals)

        return input_signals, arousals


class PhysionetPreloadDataset(Dataset):
    def __init__(self, dir='/mnt/lun1/physionet/challenge-2018/training'):
        self.dir = dir
        self.listdir = os.listdir(dir)

        self.input_signals = []
        self.arousals = []

        print('------------- STARTING DATASET LOAD ------------\n')
        start = time.time()
        for idx, folder_name in enumerate(self.listdir):

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

            # normalize
            input_signals = normalize_dataset(input_signals)

            self.input_signals.append(torch.Tensor(input_signals.values))
            self.arousals.append(torch.Tensor(arousals))

        end = time.time()
        print('Finished loading dataset in ' + str(end-start) + ' time \n')

    def __len__(self):
        return len(self.listdir)
        # return 994  # train
        # return 989  # test

    def __getitem__(self, idx):
        return self.input_signals[idx], self.arousals[idx]
