import os

import joblib
import numpy as np
import pandas as pd
import scipy.io


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

# -----------------------------------------------------------------------------
# returns a list of the training and testing file locations for easier import
# -----------------------------------------------------------------------------
# NOT USED FUNCTION
# -----------------------------------------------------------------------------
def get_files(rootDir='..'):
    header_loc, arousal_loc, signal_loc, is_training = [], [], [], []
    rootDir = '.'
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):
        if dirName != '.' and dirName != './test' and dirName != './training':
            if dirName.startswith('./training/'):
                is_training.append(True)

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '/' + fname)
                    if '-arousal.mat' in fname:
                        arousal_loc.append(dirName + '/' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '/' + fname)

            elif dirName.startswith('./test/'):
                is_training.append(False)
                arousal_loc.append('')

                for fname in fileList:
                    if '.hea' in fname:
                        header_loc.append(dirName + '/' + fname)
                    if 'mat' in fname and 'arousal' not in fname:
                        signal_loc.append(dirName + '/' + fname)

    # combine into a data frame
    data_locations = {'header':      header_loc,
                      'arousal':     arousal_loc,
                      'signal':      signal_loc,
                      'is_training': is_training
                      }

    # Convert to a data-frame
    df = pd.DataFrame(data=data_locations)

    # Split the data frame into training and testing sets.
    tr_ind = list(find(df.is_training.values))
    te_ind = list(find(df.is_training.values == False))

    training_files = df.loc[tr_ind, :]
    testing_files  = df.loc[te_ind, :]

    return training_files, testing_files

# -----------------------------------------------------------------------------
# import the outcome vector, given the file name.
# e.g. /training/tr04-0808/tr04-0808-arousal.mat
# -----------------------------------------------------------------------------
# ['data']['arousals']: (1, N)
# ['data']['sleep_stages']['nonrem1']: (N, 1)
# ['data']['sleep_stages']['nonrem2']: (N, 1)
# ['data']['sleep_stages']['nonrem3']: (N, 1)
# ['data']['sleep_stages']['rem']: (N, 1)
# ['data']['sleep_stages']['undefined']: (N, 1)
# ['data']['sleep_stages']['wake']: (N, 1)
# -----------------------------------------------------------------------------
def import_arousals(file_name):
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')
    # print(list(f['data'].keys()))
    # print(numpy.array(f['data']['sleep_stages']))
    arousals = numpy.array(f['data']['arousals'])
    return arousals


# -----------------------------------------------------------------------------
# import the measured signals
# e.g. /training/tr04-0808/tr04-0808.mat
# -----------------------------------------------------------------------------
# ['ans']['sleep_stages']['nonrem1']: (N, 1), uint8
# ['ans']['sleep_stages']['nonrem2']: (N, 1), uint8
# ['ans']['sleep_stages']['nonrem3']: (N, 1), uint8
# ['ans']['sleep_stages']['rem']: (N, 1), uint8
# ['ans']['sleep_stages']['undefined']: (N, 1), uint8
# ['ans']['sleep_stages']['wake']: (N, 1), uint8
# ['data']['arousals']: (1, N), double
# ['data']['sleep_stages']['nonrem1']: (N, 1), uint8
# ['data']['sleep_stages']['nonrem2']: (N, 1), uint8
# ['data']['sleep_stages']['nonrem3']: (N, 1), uint8
# ['data']['sleep_stages']['rem']: (N, 1), uint8
# ['data']['sleep_stages']['undefined']: (N, 1), uint8
# ['data']['sleep_stages']['wake']: (N, 1), uint8
# ['val]: (13, N), double
# -----------------------------------------------------------------------------
def import_signals(file_name):
    return np.transpose(scipy.io.loadmat(file_name)['val'])

# -----------------------------------------------------------------------------
# Take a header file as input, and returns the names of the signals
# For the corresponding .mat file containing the signals.
# -----------------------------------------------------------------------------
def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

# -----------------------------------------------------------------------------
# Get a given subject's data
# -----------------------------------------------------------------------------
def get_subject_data(arousal_file, signal_file, signal_names):
    this_arousal   = import_arousals(arousal_file)
    this_signal    = import_signals(signal_file)
    this_data      = np.append(this_signal, this_arousal, axis=1)
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data

def get_subject_data_test(signal_file, signal_names):
    this_signal    = import_signals(signal_file)
    this_data      = this_signal
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data


if __name__ == '__main__':
    folder = '../dataset/train/'
    name = 'tr03-0029/tr03-0029'
    arousals = import_arousals(folder+name+'-arousal.mat')
    header = import_signal_names(folder+name+'.hea')
    signals = import_signals(folder+name+'.mat')

    print(arousals.shape)
    print(header)
    print(signals.shape)

