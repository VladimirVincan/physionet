import gc
import os
import sys

import mat73
import numpy as np
import pandas as pd
import scipy
import torch
from dotenv import load_dotenv


def normalize_dataset(signals):
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

    normalized_signals = (signals - df_mean.iloc[0]) / df_stddev.iloc[0]

    return normalized_signals

def main():
    """
    The Challenge data repository contains two directories (training and test) which are each approximately 135 GB in size.
    Each directory contains one subdirectory per subject (e.g. training/tr03-0005).
    Each subdirectory contains signal, header, and arousal files; for example:

    1) tr03-0005.mat: a Matlab V4 file containing the signal data.
    2) tr03-0005.hea: record header file - a text file which describes the format of the signal data.
    3) tr03-0005.arousal: arousal and sleep stage annotations, in WFDB annotation format.
    4) tr03-0005-arousal.mat: a Matlab V7 structure containing a sample-wise vector with three distinct values (+1, 0, -1) where:
        a) +1: Designates arousal regions
        b) 0: Designates non-arousal regions
        c) -1: Designates regions that will not be scored

    --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

    1) Loaded with scipy.io.loadmat
       A dict of the following shape: {'val': array([[int, ...], ...], dtype=int16)}
    2) Order of signals is: F3-M2 F4-M1 C3-M2 C4-M1 O1-M2 O2-M1 E1-M2 Chin1-Chin2 ABD CHEST AIRFLOW SaO2 ECG
       All signals are int16.
       Classes 7 (Chin1-Chin2) and 11 (SaO2) are small, comprising of 576 and 28 different values respectivelly.
       Human readable, opened with myfile.read()
    3) Loaded with wfdb.rdann:
       sleep_zones = wfdb.rdann(os.path.join(os.environ['downloaded_dataset_path'], record, record), 'arousal')
       The Arousal class has two important functions:
        a) .aux_note, list with elements from the following set: {'W', '(resp_centralapnea', 'N3', '(resp_obstructiveapnea', 'resp_obstructiveapnea)', 'N2', '(arousal_rera', 'resp_centralapnea)', 'arousal_rera)', '(resp_hypopnea', 'R', 'N1', 'resp_hypopnea)'}
        b) .sample, list with timings (sample numbers)
    4) Loaded with mat73.loadmat
       It has the following structure:
       ['data']['arousals']
       ['data']['sleep_stages']['nonrem1']
       ['data']['sleep_stages']['nonrem2']
       ['data']['sleep_stages']['nonrem3']
       ['data']['sleep_stages']['rem']
       ['data']['sleep_stages']['undefined']
       ['data']['sleep_stages']['wake']
    """
    load_dotenv()

    train_df = pd.read_csv(os.path.join(os.environ['csv_path'], 'train.csv'))
    val_df   = pd.read_csv(os.path.join(os.environ['csv_path'], 'validation.csv'))
    test_df  = pd.read_csv(os.path.join(os.environ['csv_path'], 'test.csv'))

    orig_data_length = 2**23
    data_length = 2**23 // int(os.environ['stride'])

    if os.path.exists(os.path.join(os.environ['extracted_dataset_path'])):
        os.remove(os.path.join(os.environ['extracted_dataset_path'], 'trainX.dat'))
    if os.path.exists(os.path.join(os.environ['extracted_dataset_path'])):
        os.remove(os.path.join(os.environ['extracted_dataset_path'], 'trainy.dat'))

    fp_trainX = np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'trainX.dat'),dtype='float32', mode='w+', shape=(train_df.shape[0], data_length,      13))
    fp_trainy = np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'trainy.dat'),dtype='int32',   mode='w+', shape=(train_df.shape[0], data_length))
    # fp_valX =   np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'valX.dat'),  dtype='float32', mode='w+', shape=(val_df.shape[0],   data_length,      13))
    # fp_valy =   np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'valy.dat'),  dtype='int32',   mode='w+', shape=(val_df.shape[0],   orig_data_length))
    # fp_testX =  np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'testX.dat'), dtype='float32', mode='w+', shape=(test_df.shape[0],  data_length,      13))
    # fp_testy =  np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'testy.dat'), dtype='int32',   mode='w+', shape=(test_df.shape[0],  orig_data_length))

    for i, record in enumerate(train_df['Record']):
        print(i)
        arousals = mat73.loadmat(os.path.join(os.environ['downloaded_dataset_path'], record, record + '-arousal.mat'))['data']['arousals']
        signals = scipy.io.loadmat(os.path.join(os.environ['downloaded_dataset_path'], record, record + '.mat'))['val']

        # Int16 -> Float64 & Int16 -> Int32
        signals = np.array(signals).astype(np.float64)
        arousals = np.array(arousals).astype(np.int32)

        # Low-pass filter
        b, a = scipy.signal.iirfilter(int(os.environ['filter_order']), Wn=float(os.environ['Wn']), fs=200, btype="low", ftype="butter")
        for j, signal in enumerate(signals):
            signals[j] = scipy.signal.filtfilt(b, a, signal)

        # Float64 -> Float32
        signals = signals.astype(np.float32)

        # Pad signals
        pad_length = orig_data_length - signals.shape[1]
        signals = np.pad(signals, ((0, 0), (0, pad_length)), 'constant')
        arousals = np.pad(arousals, (0, pad_length), 'constant')

        # Sample with stride
        signals = signals[:, ::int(os.environ['stride'])]
        arousals = arousals[::int(os.environ['stride'])]

        # Transpose
        signals = signals.transpose()

        fp_trainX[i, :, :] = signals[:, :]
        fp_trainy[i, :] = arousals[:]

    fp_trainX.flush()
    fp_trainy.flush()
    # gc.collect()

    del fp_trainX
    del fp_trainy


def test():
    load_dotenv()

    train_df = pd.read_csv(os.path.join(os.environ['csv_path'], 'train.csv'))
    val_df   = pd.read_csv(os.path.join(os.environ['csv_path'], 'validation.csv'))
    test_df  = pd.read_csv(os.path.join(os.environ['csv_path'], 'test.csv'))

    orig_data_length = 2**23
    data_length = 2**23 // int(os.environ['stride'])

    print('---------------- test --------------')

    fp_trainX = np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'trainX.dat'), dtype='float32', mode='r', shape=(train_df.shape[0], data_length, 13))
    fp_trainy = np.memmap(os.path.join(os.environ['extracted_dataset_path'], 'trainy.dat'), dtype='int32', mode='r', shape=(train_df.shape[0], data_length))

    trainX_tensor = torch.FloatTensor(np.array(fp_trainX).astype(np.float32)).contiguous()
    trainy_tensor = torch.IntTensor(np.array(fp_trainy).astype(np.int32)).contiguous()

    print(trainX_tensor)
    print(trainy_tensor)

if __name__ == '__main__':
    main()
    test()
