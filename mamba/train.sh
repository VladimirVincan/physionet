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
python3 mamba/train_classifier.py
