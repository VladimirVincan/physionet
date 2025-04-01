import gc as garbageCollector
import os
import time

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.signal import fftconvolve, order_filter
from torch.utils.data import Dataset

import lib as phyc


class PointFiveFourDataset(Dataset):
    def __init__(self, dir='/mnt/lun1/physionet/challenge-2018/training', stride=1, pad_2e23=False):
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


def test():
    from torch.utils.data import DataLoader

    train_dataset = PointFiveFourDataset(dir=dir, num_samples=2)
    train_loader = DataLoader(train_dataset, shuffle=True)

    for batch_idx, _data in enumerate(train_loader):
        signals, arousals = _data
        print(signals.shape)
        print(arousals.shape)

        print(signals)
        print(arousals)

if __name__ == '__main__':
    test()
