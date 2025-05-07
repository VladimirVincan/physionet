import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import yaml


def import_mat(fileName):
    '''
    Read *.mat files (without -arousal).

    Attributes:
    - val (13xN): F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, E1-M2, Chin, ABD, Chest, Airflow, SaO2, ECG
    '''
    return np.transpose(scipy.io.loadmat(fileName)['val'])

def import_arousal_mat(fileName):
    '''
    Read *-arousal.mat files.

    Attributes:
    - data:
      - arousals (1xN). Values are -1 (not graded), 0 and 1.
      - sleep_stages (1x1): undefined, nonrem3, nonrem2, nonrem1, rem, wake. Values are 0 and 1, one-hot encoded.
    '''
    file =  h5py.File(fileName, 'r')
    arousals = np.array(np.array(file['data']['arousals'])).squeeze()
    stages = file['data']['sleep_stages']
    sleepStages = {
        key: np.array(stages[key]).squeeze().astype(bool)
        for key in ['undefined', 'nonrem3', 'nonrem2', 'nonrem1', 'rem', 'wake']
    }
    return arousals, sleepStages


def plot_signal_segment(input_signal: np.ndarray, arousals: np.ndarray = None, start_idx: int = 0, end_idx: int = 6000):
    """
    Plots a segment of a multichannel signal.

    Parameters:
    - input_signal: np.ndarray of shape (T, C), where T is time, and C is number of channels
    - start_idx: int, start index for plotting
    - end_idx: int, end index for plotting
    """
    if not (0 <= start_idx < end_idx <= input_signal.shape[0]):
        raise ValueError("Invalid start or end index")

    if arousals is not None and arousals.shape[0] != input_signal.shape[0]:
        raise ValueError("Output input_signal must have the same length as input signal")

    y_labels = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin', 'ABD', 'Chest', 'Airflow', 'SaO2', 'ECG']
    if arousals is not None:
        input_signal = np.concatenate((arousals[:, np.newaxis], input_signal), axis=1)
        y_labels = ['arousal'] + y_labels

    segment = input_signal[start_idx:end_idx]
    num_channels = segment.shape[1]

    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # ensure iterable for 1D

    time_indices = np.arange(start_idx, end_idx)

    for ch in range(num_channels):
        axes[ch].plot(time_indices, segment[:, ch])
        axes[ch].set_ylabel(y_labels[ch])
        axes[ch].grid(True)

    axes[-1].set_xlabel("Time Index")
    fig.suptitle(f"Signal from index {start_idx} to {end_idx}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def main():
    if len(sys.argv) == 2:
        settingsName = sys.argv[1].strip()
    else:
        print('How to use: python main.py settings_local.yaml')
        return

    with open(settingsName, 'r') as file:
        settings = yaml.safe_load(file)

    split = 'train'
    idx = 0
    starting_s = 1000
    ending_s = 2000

    starting = 200_000
    ending = 400_000

    starting = 280_000
    ending = 290_000

    folder_dir =  settings['folder_dir']
    folder_name = settings[split][idx]
    folder_path = os.path.join(folder_dir, folder_name)
    arousal_mat_path = os.path.join(folder_path, folder_name + '-arousal.mat')
    mat_path = os.path.join(folder_path, folder_name + '.mat')

    input_signal = import_mat(mat_path)
    arousals, sleep_stages = import_arousal_mat(arousal_mat_path)

    # plot_signal_segment(input_signal, arousals, 200*starting_s, 200*ending_s)
    plot_signal_segment(input_signal, arousals, starting, ending)

if __name__ == '__main__':
    main()
