#!/usr/bin/env python3
import glob
import os
import sys
import time

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from torchinfo import summary

import physionetchallenge2018_lib as phyc
from ConvFeatureExtractionModel import ConvFeatureExtractionModel
from PhysionetDataset import (PhysionetDataset, PhysionetPreloadDataset,
                              collate_fn)
from PointFiveFourModel import PointFiveFourModel
from score2018 import Challenge2018Score
from ssm import StateSpaceModel
from UNet import UNet


def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

def create_mask(labels):
    mask = labels != -1
    labels = labels[mask]
    return mask

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

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, writer):
    model.train()
    # data_len = len(train_loader.dataset)
    batch_idx = 0

    for batch_idx, _data in enumerate(train_loader):
        start = time.time()
        audio_inputs, labels = _data
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(audio_inputs)
        mask = create_mask(labels)

        # TODO: don't add negative values to loss
        # negative values -> zero
        outputs = outputs[:, :labels.shape[1], :]
        loss = criterion(outputs[mask], labels[mask])
        # print('train loss', loss)
        writer.add_scalar('Train/Loss', loss, train.writer_step)
        writer.add_scalar('Train/LR', scheduler.get_last_lr()[0], train.writer_step)
        train.writer_step += 1
        loss.backward()

        optimizer.step()
        scheduler.step()
        end = time.time()
        # print('batch : ' + str(batch_idx) + ' duration : ' + str(end-start) + ' time : ' + str(end))
        # print('batch : ' + str(batch_idx) + ' duration : ' + str(end-start))
        writer.add_scalar('Train/Batch_Time', end-start, train.writer_step)
        # pass
    writer.add_scalar('Train/Steps_in_Epoch', batch_idx, epoch)
train.writer_step=0

def validate(model, device, val_loader, criterion, epoch, writer):
    model.eval()

    score = Challenge2018Score()
    with torch.no_grad():
        for batch_idx, _data in enumerate(val_loader):
            audio_inputs, labels = _data
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)

            outputs = model(audio_inputs)
            outputs = outputs[:, :labels.shape[1], :]

            mask = create_mask(labels)
            loss = criterion(outputs[mask], labels[mask])
            writer.add_scalar('Val/Loss', loss, validate.writer_step)
            validate.writer_step += 1

            sigmoid = torch.nn.Sigmoid()
            outputs = sigmoid(outputs)

            record_name = str(batch_idx)

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs = np.squeeze(outputs)
            labels = np.squeeze(labels)

            score.score_record(labels, outputs, record_name)
            # auroc = score.record_auroc(record_name)
            # auprc = score.record_auprc(record_name)
            # print(' AUROC:%f AUPRC:%f' % (auroc, auprc))

    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    print('Training AUROC Performance (gross): %f' % auroc_g)
    print('Training AUPRC Performance (gross): %f' % auprc_g)
    writer.add_scalar('auroc/test', auroc_g, epoch)
    writer.add_scalar('auprc/test', auprc_g, epoch)
validate.writer_step=0

def test(model, device, test_loader):
    model.eval()

    score = Challenge2018Score()
    with torch.no_grad():
        for batch_idx, _data in enumerate(test_loader):
            audio_inputs, labels = _data
            audio_inputs = audio_inputs.to(device)
            labels = labels.to(device)

            outputs = model(audio_inputs)
            sigmoid = torch.nn.Sigmoid()
            outputs = sigmoid(outputs)

            record_name = str(batch_idx)

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs = np.squeeze(outputs)
            labels = np.squeeze(labels)

            score.score_record(labels, outputs, record_name)
            auroc = score.record_auroc(record_name)
            auprc = score.record_auprc(record_name)
            print(' AUROC:%f AUPRC:%f' % (auroc, auprc))

    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    print('Training AUROC Performance (gross): %f' % auroc_g)
    print('Training AUPRC Performance (gross): %f' % auprc_g)


def main():
    init()

    config_file = 'mamba/config.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    writer = SummaryWriter(log_dir=config['summary_writer'])

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")

    # [(x, y, z)]:
    # x - out channels
    # y - kernel size
    # z - stride
    feature_enc_layers = "[(128, 1, 1)]"
    feature_mamba_layers = "[(128)]"  # TODO
    # feature_enc_layers = eval(feature_enc_layers)
    print('-------------------------- MODEL -------------------------\n', flush=True)
    # model = PointFiveFourModel(13)
    model = StateSpaceModel(feature_enc_layers, feature_mamba_layers)
    # model = ConvFeatureExtractionModel(feature_enc_layers)
    # model = UNet(13, 32, 32, [5, 2, 5], 3)
    print('-------------------------- CUDA -------------------------\n', flush=True)
    model.to(device)
    # record = '../challenge-2018/training/tr03-0005/tr03-0005'

    # batch, length, dimension
    # print('-------------------------- SUMMARY -------------------------\n', flush=True)
    # # TODO: check https://discuss.pytorch.org/t/why-does-the-size-of-forward-backward-pass-differ-when-using-a-single-class-for-a-model-and-partitioning-the-model-using-different-classes-and-later-accumulating-it/185294
    summary(model,
            (1, 100000, 13),
            device=device,
            verbose=1,
            depth=5,
            col_names=['input_size',
                       'output_size',
                       "num_params",
                       "params_percent",
                       "kernel_size",
                       "mult_adds",
                       "trainable",])

    train_dataset = PhysionetDataset(config['train_dataset'], 16)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=8)

    # TODO: is batch_size=1 needed?
    val_dataset = PhysionetDataset(config['val_dataset'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    print('-------------------------- LOADER LENGTH -------------------------\n', flush=True)
    print('Train Loader length: ' + str(len(train_loader)), flush=True)
    print('Val Loader length: ' + str(len(val_loader)), flush=True)

    epochs = 125

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0])).to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = 4e-3,
        pct_start = 0.05,
        steps_per_epoch=int(len(train_loader)),
        epochs=epochs,
        anneal_strategy='linear'
    )

    print('-------------------------- EPOCHS -------------------------\n', flush=True)
    iter_meter = IterMeter()
    for epoch in range(1, epochs + 1):
        print('============ EPOCH: ' + str(epoch) + ' ============', flush=True)
        train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, writer)
        validate(model, device, val_loader, criterion, epoch, writer)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Save this algorithm for submission to Physionet Challenge:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('-------------------------- SAVE MODEL -------------------------\n', flush=True)
    joblib.dump(model, config['model_name'])
    writer.flush()

    print('============ TESTING ============')
    # Using PhysionetDataset instead of PhysionetPreloadDataset to reduce the memory consumption
    test_dataset = PhysionetDataset(config['test_dataset'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    test(model, device, test_loader)

    writer.flush()


if __name__ == '__main__':
    main()
