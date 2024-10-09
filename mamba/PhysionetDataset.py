import gc as garbageCollector
import os
import time

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.signal import fftconvolve, order_filter
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
    def __init__(self, dir='/mnt/lun1/physionet/challenge-2018/training', stride=1, train=True, order=4, Wn=25.0):
        self.dir = dir
        self.listdir = os.listdir(dir)
        self.stride = stride
        self.train = train
        self.order = order
        self.Wn = Wn

    def __len__(self):
        return len(self.listdir) * self.stride
        # return 994  # train
        # return 989  # test

    def __getitem__(self, idx):
        start = idx % self.stride
        idx = idx // self.stride

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

        # Low pass
        b, a = scipy.signal.iirfilter(self.order, Wn=self.Wn, fs=200, btype="low", ftype="butter")
        for label, signal in input_signals.items():
            input_signals[label] = scipy.signal.filtfilt(b, a, signal)  # TODO: filfilt or lfilter?

        # Convert to 32 bits
        input_signals = input_signals.astype(np.float32)
        arousals = arousals.astype(np.float32)

        # Sample every nth row (stride)
        if self.stride > 1:
            input_signals = input_signals.iloc[start::self.stride, :]

            if self.train == True:
                arousals = arousals[start::self.stride, :]

        input_signals = torch.Tensor(input_signals.values)
        arousals = torch.Tensor(arousals)

        return input_signals, arousals


class PhysionetPreloadDataset(Dataset):
    """
    No need to use anymore since https://discuss.pytorch.org/t/simultaneously-preprocess-a-batch-on-cpu-and-run-forward-backward-on-gpu/161987/2
    """
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


class PointFiveFourDataset(Dataset):
    def __init__(self, dir='/mnt/lun1/physionet/challenge-2018/training', stride=1):
        self.dir = dir
        self.listdir = os.listdir(dir)
        self.stride = stride

    def __len__(self):
        return len(self.listdir)

    def __getitem__(self, idx):
        start = idx % self.stride
        idx = idx // self.stride

        record_name = os.path.join(self.dir, self.listdir[idx], self.listdir[idx])
        header_file = record_name + '.hea'
        signal_file = record_name + '.mat'
        arousal_file = record_name + '-arousal.mat'

        # Get the signal names from the header file
        signal_names, Fs, n_samples = phyc.import_signal_names(header_file)
        signal_names = list(np.append(signal_names, 'arousals'))

        this_data = phyc.get_subject_data(arousal_file, signal_file, signal_names)
        input_signals = this_data.loc[:, this_data.columns!='arousals']
        arousals = this_data.get(['arousals']).values

        signals = input_signals.to_numpy()

        # Keep all channels except ECG
        keepChannels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        # Apply antialiasing FIR filter to each channel and downsample to 50Hz
        filtCoeff = np.array([0.00637849379422531, 0.00543091599801427, -0.00255136650039784, -0.0123109503066702,
                              -0.0137267267561505, -0.000943230632358082, 0.0191919895027550, 0.0287148886882440,
                              0.0123598773891149, -0.0256928886371578, -0.0570987715759348, -0.0446385294777459,
                              0.0303553522906817, 0.148402006671856, 0.257171285176269, 0.301282456398562,
                              0.257171285176269, 0.148402006671856, 0.0303553522906817, -0.0446385294777459,
                              -0.0570987715759348, -0.0256928886371578, 0.0123598773891149, 0.0287148886882440,
                              0.0191919895027550, -0.000943230632358082, -0.0137267267561505, -0.0123109503066702,
                              -0.00255136650039784, 0.00543091599801427, 0.00637849379422531])

        for n in range(signals.shape[1]):
            signals[::, n] = np.convolve(signals[::, n], filtCoeff, mode='same')

        signals = signals[0::4, keepChannels]
        arousals = arousals[0::4]

        garbageCollector.collect()

        # Scale SaO2 to sit between -0.5 and 0.5, a good range for input to neural network
        signals[::, 11] += -32768.0
        signals[::, 11] /= 65535.0
        signals[::, 11] -= 0.5

        # Normalize all the other channels by removing the mean and the rms in an 18 minute rolling window, using fftconvolve for computational efficiency
        # 18 minute window is used because because baseline breathing is established in 2 minute window according to AASM standards.
        # Normalizing over 18 minutes ensure a 90% overlap between the beginning and end of the baseline window
        kernel_size = (50*18*60)+1

        # Remove DC bias and scale for FFT convolution
        center = np.mean(signals, axis=0)
        scale = np.std(signals, axis=0)
        scale[scale == 0] = 1.0
        signals = (signals - center) / scale

        # Compute and remove moving average with FFT convolution
        center = np.zeros(signals.shape)
        for n in range(signals.shape[1]):
            center[::, n] = fftconvolve(signals[::, n], np.ones(shape=(kernel_size,))/kernel_size, mode='same')

        # Exclude SAO2
        center[::, 11] = 0.0
        center[np.isnan(center) | np.isinf(center)] = 0.0
        signals = signals - center

        # Compute and remove the rms with FFT convolution of squared signal
        scale = np.ones(signals.shape)
        for n in range(signals.shape[1]):
            temp = fftconvolve(np.square(signals[::, n]), np.ones(shape=(kernel_size,))/kernel_size, mode='same')

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
        signals = signals / scale

        garbageCollector.collect()

        # Convert to 32 bits
        signals = signals.astype(np.float32)

        # Convert to torch tensor
        signals = torch.from_numpy(signals)
        arousals = torch.from_numpy(signals)

        return signals, arousals

def collate_fn(data: list[tuple[torch.Tensor, torch.Tensor]]):
    # https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
    # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
    tensors, targets = zip(*data)
    features = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1.0)
    return features, labels


def collate_fn_numpy2tensor(data: list[tuple[torch.Tensor, torch.Tensor]]):
    # https://stackoverflow.com/questions/65279115/how-to-use-collate-fn-with-dataloaders
    # https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html
    # https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
    tensors, targets = zip(*data)
    features = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
    labels = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-1.0)
    return features, labels
