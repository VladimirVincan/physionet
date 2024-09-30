import os
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import physionetchallenge2018_lib as phyc


def normalize_dataset(input_signals):
    data_max = {
        'F3-M2':       [8646.0],
        'F4-M1':       [8774.0],
        'C3-M2':       [8198.0],
        'C4-M1':       [8110.0],
        'O1-M2':       [7522.0],
        'O2-M1':       [8367.0],
        'E1-M2':       [8013.0],
        'Chin1-Chin2': [6134.0],
        'ABD':         [8533.0],
        'CHEST':       [8520.0],
        'AIRFLOW':     [8394.0],
        'SaO2':        [32443.0],
        'ECG':         [5022.0]
    }

    data_min = {
        'F3-M2':       [-8440.0],
        'F4-M1':       [-8229.0],
        'C3-M2':       [-7753.0],
        'C4-M1':       [-8093.0],
        'O1-M2':       [-7710.0],
        'O2-M1':       [-7971.0],
        'E1-M2':       [-8718.0],
        'Chin1-Chin2': [-5947.0],
        'ABD':         [-8538.0],
        'CHEST':       [-8480.0],
        'AIRFLOW':     [-8410.0],
        'SaO2':        [-32444.0],
        'ECG':         [-4984.0]
    }

    data_sum = {
        'F3-M2':       [ 68188554],
        'F4-M1':       [-116538705],
        'C3-M2':       [ 152193279],
        'C4-M1':       [ 37612813],
        'O1-M2':       [ 33657544],
        'O2-M1':       [-40815548],
        'E1-M2':       [ 156439760],
        'Chin1-Chin2': [-463259],
        'ABD':         [-3446192437],
        'CHEST':       [-1723891404],
        'AIRFLOW':     [ 1653144237],
        'SaO2':        [ 304333928543669],
        'ECG':         [-28810712]
    }

    data_count = {
        'F3-M2':       [10988088400],
        'F4-M1':       [10988088400],
        'C3-M2':       [10988088400],
        'C4-M1':       [10988088400],
        'O1-M2':       [10988088400],
        'O2-M1':       [10988088400],
        'E1-M2':       [10988088400],
        'Chin1-Chin2': [10988088400],
        'ABD':         [10988088400],
        'CHEST':       [10988088400],
        'AIRFLOW':     [10988088400],
        'SaO2':        [10988088400],
        'ECG':         [10988088400]
    }

    data_stddev = {
        'F3-M2':       [2.044852],
        'F4-M1':       [1.917928],
        'C3-M2':       [1.831256],
        'C4-M1':       [2.135969],
        'O1-M2':       [1.274364],
        'O2-M1':       [1.301722],
        'E1-M2':       [1.884008],
        'Chin1-Chin2': [0.512828],
        'ABD':         [14.808272],
        'CHEST':       [19.670875],
        'AIRFLOW':     [21.859140],
        'SaO2':        [539.818647],
        'ECG':         [1.708024]
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

        # Normalize
        input_signals = normalize_dataset(input_signals)

        # Convert to 32 bits
        input_signals = input_signals.astype(np.float32)
        arousals = arousals.astype(np.float32)

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
            # this_data = this_data.iloc[:100001, :]

            input_signals = this_data.loc[:, this_data.columns!='arousals']
            arousals = this_data.get(['arousals']).values

            # Normalize
            input_signals = normalize_dataset(input_signals)

            # Convert to 32 bits
            input_signals = input_signals.astype(np.float32)
            arousals = arousals.astype(np.float32)

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
