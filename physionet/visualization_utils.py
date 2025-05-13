import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import wfdb
import yaml

from data_reader_utils import (get_filepaths_dict, import_arousal,
                               import_arousal_mat, import_mat)


def find_positions(string_list, target_string):
    return [i for i, s in enumerate(string_list) if s == target_string]

def get_respiratory_signal(annotation, signal_name, num_samples):
    signal = np.zeros(shape=(num_samples))

    start_string_name = '(' + signal_name
    end_string_name = signal_name + ')'

    start_positions = find_positions(annotation['aux_note'], start_string_name)
    end_positions = find_positions(annotation['aux_note'], end_string_name)

    start_timestamps = np.array(annotation['sample'])[start_positions]
    end_timestamps = np.array(annotation['sample'])[end_positions]

    for start, end in zip(start_timestamps, end_timestamps):
        signal[start:end] = 1

    return signal

def find_next_positions(string_list, target_string, start_positions):
    allowed_next_strings = ['W', 'N1', 'N2', 'N3', 'R']
    allowed_next_strings.remove(target_string)

    next_positions = []
    for i in start_positions:
        j = i + 1
        while (j < len(string_list)) and (not (string_list[j] in allowed_next_strings)):
            j = j + 1
        if j != len(string_list):
            next_positions.append(j)
    return next_positions

def get_sleep_stage_signal(annotation, signal_name, num_samples):
    signal = np.zeros(shape=(num_samples))

    start_positions = find_positions(annotation['aux_note'], signal_name)
    end_positions = find_next_positions(annotation['aux_note'], signal_name, start_positions)

    start_timestamps = np.array(annotation['sample'])[start_positions]
    end_timestamps = np.array(annotation['sample'])[end_positions]

    # there should be
    if len(end_timestamps) < len(start_timestamps):
        assert len(end_timestamps) == len(start_timestamps) - 1
        end_timestamps  = np.append(end_timestamps, num_samples)
    assert len(end_timestamps) == len(start_timestamps)

    for start, end in zip(start_timestamps, end_timestamps):
        signal[start:end] = 1

    return signal

def make_xlim_callback(axs, input_signals, fig):
    def on_xlim_change(event_ax):
        new_xlim = event_ax.get_xlim()
        start = int(max(0, np.floor(new_xlim[0])))
        end = int(min(input_signals.shape[0], np.ceil(new_xlim[1])))
        # print(new_xlim)
        # print('xlim: ' + str(start) + ' ' + str(end))
        for i, ax in enumerate(axs):
            # print(input_signals.shape)
            y_segment = input_signals[start:end, i]
            ymin, ymax = np.min(y_segment), np.max(y_segment)
            # print(ymin, ymax)
            diff = ymax - ymin
            ymax += diff*0.1
            ymin -= diff*0.1
            ax.set_ylim(ymin, ymax)
        fig.canvas.draw_idle()
    return on_xlim_change

def plot_signal_segment(input_signal: np.ndarray, arousals: np.ndarray = None, arousals_dict = None, start_idx: int = 0, end_idx: int = 6000):
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

    for key, value in arousals_dict.items():
        input_signal = np.concatenate((value[:, np.newaxis], input_signal), axis=1)
        y_labels = [key] + y_labels

    if arousals is not None:
        input_signal = np.concatenate((arousals[:, np.newaxis], input_signal), axis=1)
        y_labels = ['arousal'] + y_labels

    segment = input_signal[start_idx:end_idx]
    num_channels = segment.shape[1]

    fig, axes = plt.subplots(num_channels, 1, figsize=(15, 2*num_channels), sharex=True)
    if num_channels == 1:
        axes = [axes]  # ensure iterable for 1D

    time_indices = np.arange(start_idx, end_idx)

    callback = make_xlim_callback(axes, input_signal, fig)
    for ch in range(num_channels):
        axes[ch].plot(time_indices, segment[:, ch])
        axes[ch].set_ylabel(y_labels[ch])
        axes[ch].grid(True)
        axes[ch].callbacks.connect('xlim_changed', callback)

    axes[-1].set_xlabel("Time Index")
    fig.suptitle(f"Signal from index {start_idx} to {end_idx}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

def find_regions_of_ones(signal: np.ndarray):
    """
    Finds the start and end indices of contiguous regions where the signal == 1.

    Parameters:
        signal: np.ndarray of shape (N,) or (N, 1)

    Returns:
        List of tuples (start_idx, end_idx), where signal[start_idx:end_idx] == 1
    """
    # Flatten in case input is (N, 1)
    signal = signal.flatten()

    # Identify where signal == 1
    is_one = signal == 1

    # Find changes in the signal
    diff = np.diff(is_one.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    starts = starts/200.
    ends = ends/200.

    # Handle edge case: signal starts with 1
    if is_one[0]:
        starts = np.insert(starts, 0, 0)
    # Handle edge case: signal ends with 1
    if is_one[-1]:
        ends = np.append(ends, len(signal))

    regions = list(zip(starts, ends))
    return regions


def main():
    """
    View raw data files without preprocessing.
    """
    if len(sys.argv) == 2:
        settings_name = sys.argv[1].strip()
    else:
        print('How to use: python visualization_utils.py settings_local.yaml')
        return

    with open(settings_name, 'r') as file:
        settings = yaml.safe_load(file)

    filepaths_dict = get_filepaths_dict(settings)
    input_signals = import_mat(filepaths_dict['mat_path'])
    output_signal, _ = import_arousal_mat(filepaths_dict['arousal_mat_path'])
    num_samples = output_signal.shape[0]  # (N, 1) -> N
    arousal_signals = import_arousal(filepaths_dict['wfdb_path'], num_samples)

    starting = 0  # 100_000
    ending = num_samples  # 4_000_000

    plot_signal_segment(input_signals, output_signal, arousal_signals, starting, ending)

if __name__ == '__main__':
    main()
