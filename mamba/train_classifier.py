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
from PhysionetDataset import PhysionetDataset, PhysionetPreloadDataset
from score2018 import Challenge2018Score
from ssm import StateSpaceModel


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

def train(model, device, train_loader, criterion, optimizer, scheduler, epoch, iter_meter, writer):
    model.train()
    # data_len = len(train_loader.dataset)

    for batch_idx, _data in enumerate(train_loader):
        start = time.time()
        audio_inputs, labels = _data
        audio_inputs = audio_inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(audio_inputs)

        # TODO: don't add negative values to loss
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

def validate(model, device, val_loader, epoch, writer):
    model.eval()

    score = Challenge2018Score()
    with torch.no_grad():
        for batch_idx, _data in enumerate(val_loader):
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

    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    print('Training AUROC Performance (gross): %f' % auroc_g)
    print('Training AUPRC Performance (gross): %f' % auprc_g)
    writer.add_scalar('auroc/test', auroc_g, epoch)
    writer.add_scalar('auprc/test', auprc_g, epoch)

def test(model, device, test_loader):
    model.eval()

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

    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()
    print('Training AUROC Performance (gross): %f' % auroc_g)
    print('Training AUPRC Performance (gross): %f' % auprc_g)


def main():
    init()

    config_file = 'mamba/config_fmle.yaml'
    config_file = 'mamba/config_local.yaml'
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    writer = SummaryWriter(log_dir=config['summary_writer'])

    use_cuda = torch.cuda.is_available()
    torch.manual_seed(42)
    device = torch.device("cuda" if use_cuda else "cpu")

    print('-------------------------- STARTING -------------------------\n', flush=True)
    # [(x, y, z)]:
    # x - out channels
    # y - kernel size
    # z - stride
    feature_enc_layers = "[(32, 1, 1)]"
    feature_mamba_layers = "[(32)]"  # TODO
    # feature_enc_layers = eval(feature_enc_layers)
    print('-------------------------- MODEL -------------------------\n', flush=True)
    model = StateSpaceModel(feature_enc_layers, feature_mamba_layers)
    # model = ConvFeatureExtractionModel(feature_enc_layers)
    print('-------------------------- CUDA -------------------------\n', flush=True)
    model.to(device)
    # record = '../challenge-2018/training/tr03-0005/tr03-0005'

    # batch, length, dimension
    # print('-------------------------- SUMMARY -------------------------\n', flush=True)
    # # TODO: check https://discuss.pytorch.org/t/why-does-the-size-of-forward-backward-pass-differ-when-using-a-single-class-for-a-model-and-partitioning-the-model-using-different-classes-and-later-accumulating-it/185294
    # summary(model,
    #         (1, 8*60*60*200, 13),
    #         device=device,
    #         verbose=1,
    #         depth=7,
    #         col_names=['input_size',
    #                    'output_size',
    #                    "num_params",
    #                    "params_percent",
    #                    "kernel_size",
    #                    "mult_adds",
    #                    "trainable",])

    print('-------------------------- TRAIN -------------------------\n', flush=True)
    train_dataset = PhysionetPreloadDataset(config['train_dataset'])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    all_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    all_labels = all_labels[1]
    mask = all_labels != -1
    all_labels = all_labels[mask]
    unique_labels, counts = torch.unique(all_labels, return_counts=True)

    print('-------------------------- VAL -------------------------\n', flush=True)
    val_dataset = PhysionetPreloadDataset(config['val_dataset'])
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    print('-------------------------- LOADER LENGTH -------------------------\n', flush=True)
    print('Train Loader length: ' + str(len(train_loader)), flush=True)
    print('Val Loader length: ' + str(len(val_loader)), flush=True)

    epochs = 100

    print('-------------------------- OPT CRI SCH -------------------------\n', flush=True)
    optimizer = optim.AdamW(model.parameters())
    criterion = nn.BCEWithLogitsLoss(pos_weight=10.0) #.to(device)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = 1e-5,
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
        validate(model, device, val_loader, epoch, writer)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Save this algorithm for submission to Physionet Challenge:
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print('-------------------------- SAVE MODEL -------------------------\n', flush=True)
    joblib.dump(model, config['model_name'])
    writer.flush()

    print('============ TESTING ============')
    # Using PhysionetDataset instead of PhysionetPreloadDataset to reduce the memory consumption
    test_dataset = PhysionetDataset(config['test_dataset'])
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    test(model, device, test_loader)

    writer.flush()


if __name__ == '__main__':
    main()
