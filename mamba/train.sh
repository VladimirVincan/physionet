#!/bin/bash

pwd
echo "--------------------------------"
ls -alR
echo "--------------------------------"
ls -al
echo "--------------------------------"

pip install tensorboard
pip install torchinfo
pip install h5py
pip install einops
pip install transformers
python3 mamba/train_classifier.py
