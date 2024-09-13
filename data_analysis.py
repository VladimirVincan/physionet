import os

import joblib
import numpy as np
import pandas as pd
import scipy.io


def main():

    ann = wfdb.rdann('sample-data/100', 'atr', sampto=300000)
    pass


if __name__ == '__main__':
    main()
