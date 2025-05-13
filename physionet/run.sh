#!/bin/bash

pip install tensorboard
pip install torchinfo
pip install h5py
pip install einops
pip install transformers
pip install dotenv
pip install wfdb
pip isntall matplotlib
pip install scikit-learn

python3 physionet/main.py physionet/settings.yaml
