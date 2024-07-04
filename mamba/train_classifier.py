#!/usr/bin/env python3
import glob
import os
import sys

import joblib
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torchinfo import summary

import physionetchallenge2018_lib as phyc
from score2018 import Challenge2018Score
from ConvFeatureExtractionModel import ConvFeatureExtractionModel
from PhysionetDataset import PhysionetDataset, PhysionetPreloadDataset
from ssm import StateSpaceModel

writer = SummaryWriter(log_dir='/out/in')


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
        start = time.time()
        audio_inputs, labels = _data
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(audio_inputs)

        # negative values -> zero
        labels = torch.clamp(labels, min=0)
        loss = criterion(outputs, labels)
        # print('train loss', loss)
        writer.add_scalar('Loss/train', loss, train.writer_step)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], train.writer_step)
        train.writer_step += 1
        loss.backward()

        optimizer.step()
        scheduler.step()
        end = time.time()
        print('batch : ' + str(batch_idx) + ' duration : ' + str(end-start) + ' time : ' + str(end))
        writer.add_scalar('time/batch', end-start, train.writer_step)
        # pass
train.writer_step=0

def test(model, device, test_loader, criterion, epoch, iter_meter):
    model.eval()
    test_loss = 0

    score = Challenge2018Score()
    with torch.no_grad():
        for batch_idx, _data in enumerate(test_loader):
            audio_inputs, labels = _data
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)

            outputs = model(audio_inputs)

            record_name = str(batch_idx)

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs = np.squeeze(outputs)
            labels = np.squeeze(labels)
            score.score_record(labels, outputs, record_name)
            auroc = score.record_auroc(record_name)
            auprc = score.record_auprc(record_name)
            print(' AUROC:%f AUPRC:%f' % (auroc, auprc))

    print()
    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    print('Training AUROC Performance (gross): %f' % auroc_g)
    print('Training AUPRC Performance (gross): %f' % auprc_g)
    writer.add_scalar('auroc/test', auroc_g, epoch)
    writer.add_scalar('auprc/test', auprc_g, epoch)

def preprocess_record(model_name):
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

    feature_enc_layers = "[(32, 1, 1)]"
    feature_mamba_layers = "[(32)]"  # TODO
    # feature_enc_layers = eval(feature_enc_layers)
    model = StateSpaceModel(feature_enc_layers, feature_mamba_layers)
    # model = ConvFeatureExtractionModel(feature_enc_layers, mode='layer_norm')
    model.to(device)

    # batch, length, dimension
    summary(model, (1, 8*60*60*200, 13), device='cuda')

    # train_dataset = PhysionetDataset('/home/bici/physionet/challenge-2018/training')
    # train_dataset = PhysionetDataset()
    train_dataset = PhysionetPreloadDataset('/data/physionet/challenge-2018/training')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print('Train Loader length: ' + str(len(train_loader)))

    epochs = 100

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.BCELoss() #.to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = 1e-5,
        pct_start = 0.05,
        steps_per_epoch=int(len(train_loader)),
        epochs=epochs,
        anneal_strategy='linear'
    )

    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        print('============ EPOCH: ' + str(epoch) + ' ============')
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter)

    print('============ TESTING ============')
    test_dataset = PhysionetDataset('/data/physionet/challenge-2018/training')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test(model, device, test_loader, criterion, epoch, iter_meter)
    writer.flush()

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Save this algorithm for submission to Physionet Challenge:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # model_file = 'models/%s_model.pkl' % os.path.basename(record_name)
    joblib.dump(model, model_name)

def finish():
    pass

if __name__ == '__main__':
    init()
    # train()
    # record = '../challenge-2018/training/tr03-0005/tr03-0005'
    preprocess_record('test_model')
    # finish()
    writer.close()
