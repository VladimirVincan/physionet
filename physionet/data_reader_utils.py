import os
import sys

import h5py
import numpy as np
import scipy.io
import wfdb
import yaml

# =================================== filepaths ===================================

def get_filepaths_dict(settings, split=None, idx=None):
    """
    input: path to folder with all subfolders
    output: file/path names to util functions
    """
    if split == None or idx == None:
        split = settings['split']
        idx = settings['idx']
    folder_dir =  settings['folder_dir']
    folder_name = settings[split][idx]

    folder_path = os.path.join(folder_dir, folder_name)
    wfdb_path = os.path.join(folder_path, folder_name)

    arousal_mat_path = os.path.join(folder_path, folder_name + '-arousal.mat')
    mat_path = os.path.join(folder_path, folder_name + '.mat')
    hea_path = os.path.join(folder_path, folder_name + '.hea')
    arousal_path = os.path.join(folder_path, folder_name + '.arousal')
    outputs_path = os.path.join(folder_path, folder_name + '.npy')

    filepaths_dict = {
        'folder_name': folder_name,
        'wfdb_path': wfdb_path,
        'mat_path': mat_path,
        'arousal_mat_path': arousal_mat_path,
        'hea_path': hea_path,
        'arousal_path': arousal_path,
        'outputs_path': outputs_path
    }

    return filepaths_dict

# =================================== ! filepaths ! ===================================

# =================================== Load wfdb ===================================

def read_ann(wfdb_path):
    """
    record_name: folder_name
    extension: 'arousal'
    sample: array() of timestamps, 1xM where M is number of events
    symbol: ?
    subtype: array() of event codes, 1xM where M is number of events
    aux_note: array() of event names, 1xM where M is number of events
    fs: 200
    label_store: ?, None
    description: ?, None
    custom_labels: ?, None
    contained_labels: ?, None
    ann_length: M, number of events
    """
    ann = wfdb.rdann(wfdb_path, 'arousal')
    ann = ann.__dict__
    return ann

def read_record(wfdb_path):
    """
    record_name: folder_name
    n_sig: 13
    fs: 200
    counter_freq: ?, None
    base_counter: ?, None
    sig_len: ~5 million
    base_time: ?, None
    base_date: ?, None
    comments: ?, []
    sig_name: ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
    p_signal: array() of n_sig x sig_len
    d_signal: None
    e_p_signal: None
    e_d_signal: None
    file_name: [record_name...]
    fmt: ?, [16...]['16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16', '16']
    samps_per_frame: [1...][1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    skew: [None...][None, None, None, None, None, None, None, None, None, None, None, None, None]
    byte_offset: [24...][24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24]
    adc_gain: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 655.35, 1000.0]
    baseline: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -32768, 0]
    units: ['uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', 'uV', '%', 'mV']
    adc_res: [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
    adc_zero: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    init_value: [-9, 5, -5, 9, 5, 12, -21, 16, -41, -2, 39, 30496, -53]
    checksum: [139, 1793, 1290, -357, 248, -96, -1068, -741, -31565, -21968, 2879, 9703, -1118]
    block_size: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    record = wfdb.rdrecord(wfdb_path)
    record = record.__dict__
    return record

# =================================== ! Load wfdb ! ===================================

# =================================== Load individual data files ===================================
"""
import_mat
import_arousal_mat
import_arousal

import_individual_arousal_signal
"""

def import_mat(fileName):
    '''
    Read *.mat files (without -arousal).

    Attributes:
    - val (13xN): F3-M2, F4-M1, C3-M2, C4-M1, O1-M2, O2-M1, E1-M2, Chin, ABD, Chest, Airflow, SaO2, ECG
    '''
    return np.transpose(scipy.io.loadmat(fileName)['val'])

def get_input_signal_position(signal_name):
    signal_names = ['F3-M2', 'F4-M1', 'C3-M2', 'C4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'Chin1-Chin2', 'ABD', 'CHEST', 'AIRFLOW', 'SaO2', 'ECG']
    position = signal_names.index(signal_name)
    return position

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

def import_individual_arousal_signal(folder_path, num_samples, signal_name):
    annotation = wfdb.rdann(folder_path, 'arousal')
    annotation = annotation.__dict__

    if signal_name in ['resp_obstructiveapnea', 'resp_centralapnea', 'resp_mixedapnea', 'resp_hypopnea', 'arousal_rera']:
        signal = get_respiratory_signal(annotation, signal_name, num_samples)
    elif signal_name in ['W', 'N1', 'N2', 'N3', 'R']:
        signal = get_sleep_stage_signal(annotation, signal_name, num_samples)
    else:
        raise Exception('Unknown signal name')

    return {signal_name: signal}

def import_arousal(folder_path, num_samples):
    annotation = wfdb.rdann(folder_path, 'arousal')
    annotation = annotation.__dict__

    resp_obstructiveapnea = get_respiratory_signal(annotation, 'resp_obstructiveapnea', num_samples)
    resp_centralapnea = get_respiratory_signal(annotation, 'resp_centralapnea', num_samples)
    resp_mixedapnea = get_respiratory_signal(annotation, 'resp_mixedapnea', num_samples)
    resp_hypopnea = get_respiratory_signal(annotation, 'resp_hypopnea', num_samples)
    resp_arousalrera = get_respiratory_signal(annotation, 'arousal_rera', num_samples)

    # TODO: ukloniti posto je implementirano vec u -arousal.mat fajlu
    wake_signal = get_sleep_stage_signal(annotation, 'W', num_samples)
    nrem1_signal = get_sleep_stage_signal(annotation, 'N1', num_samples)
    nrem2_signal = get_sleep_stage_signal(annotation, 'N2', num_samples)
    nrem3_signal = get_sleep_stage_signal(annotation, 'N3', num_samples)
    rem_signal = get_sleep_stage_signal(annotation, 'R', num_samples)

    arousal_dict = {
        'obst': resp_obstructiveapnea,
        'cent': resp_centralapnea,
        'mix': resp_mixedapnea,
        'hypo': resp_hypopnea,
        'rera': resp_arousalrera,
        'wake': wake_signal,
        'nrem1': nrem1_signal,
        'nrem2': nrem2_signal,
        'nrem3': nrem3_signal,
        'rem': rem_signal
    }

    return arousal_dict

# =================================== ! Load individual data files ! ===================================


def main():
    if len(sys.argv) == 2:
        settingsName = sys.argv[1].strip()
    else:
        print('How to use: python main.py settings_local.yaml')
        return

    with open(settingsName, 'r') as file:
        settings = yaml.safe_load(file)

    filepaths_dict = get_filepaths_dict(settings)
    print(filepaths_dict)
    # input_data, output_data = _data(filepaths_dict)
    # print(read_ann(filepaths_dict['wfdb_path']))
    # print(read_record(filepaths_dict['wfdb_path']))



if __name__ == '__main__':
    main()

