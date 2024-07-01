#!/usr/bin/env python3
import glob
import os
import sys

import joblib
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torchinfo import summary

import physionetchallenge2018_lib as phyc
from ConvFeatureExtractionModel import ConvFeatureExtractionModel
from PhysionetDataset import PhysionetDataset
from ssm import StateSpaceModel

writer = SummaryWriter(log_dir='./log')


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val

def init():
    # Create the 'models' subdirectory and delete any existing model files
    try:
        os.mkdir('models')
    except OSError:
        pass
    for f in glob.glob('models/*_model.pkl'):
        os.remove(f)

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter):
    model.train()
    # data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        audio_inputs, labels = _data

        optimizer.zero_grad()

        outputs = model(audio_inputs)

        # print(outputs.shape)
        loss = criterion(outputs, labels)
        # print('train loss', loss)
        print('batch idx: ' + str(batch_idx))
        writer.add_scalar('Loss/train', loss, train.writer_step)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], train.writer_step)
        train.writer_step += 1
        loss.backward()

        optimizer.step()
        scheduler.step()
        # pass
train.writer_step=0

def preprocess_record(record_name):
    # ----------------------------------------------------------------------
    # Generate the Features for the classificaition model - variance of SaO2
    # ----------------------------------------------------------------------

    # For the baseline, let's only look at how SaO2 might predict arousals


    # ---------------------------------------------------------------------
    # Train a (multi-class) Logistic Regression classifier
    # ---------------------------------------------------------------------
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")

    # model = StateSpaceModel()
    feature_enc_layers = "[(1024, 13, 1)]"
    feature_enc_layers = eval(feature_enc_layers)
    model = ConvFeatureExtractionModel(feature_enc_layers, mode='layer_norm')
    model.to(device)

    # batch, length, dimension
    summary(model, (1, 8*60*60*200, 13), device='cuda')

    train_dataset = PhysionetDataset('/home/bici/physionet/challenge-2018/training')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print('Train Loader length: ' + str(len(train_loader)))

    epochs = 100

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.BCELoss #.to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = 1e-4,
        steps_per_epoch=int(len(train_loader)),
        epochs=epochs,
        anneal_strategy='linear'
    )

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)
        # test(model, device, test_loader, criterion, epoch, iter_meter)
        writer.flush()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Save this algorithm for submission to Physionet Challenge:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    model_file = 'models/%s_model.pkl' % os.path.basename(record_name)
    # joblib.dump(my_classifier, model_file)

def finish():
    pass

if __name__ == '__main__':
    init()
    # train()
    record = '../challenge-2018/training/tr03-0005/tr03-0005'
    preprocess_record(record)
    # finish()
