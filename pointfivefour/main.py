#!/usr/bin/env python3


import argparse
import copy
import csv
import glob
import os
import sys
import time

import joblib
import numpy as np
import physionetchallenge2018_lib as phyc
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from PhysionetDataset import (PhysionetDataset, PhysionetPreloadDataset,
                              collate_fn)
from score2018 import Challenge2018Score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from UNet import UNet

from model import PointFiveFourModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Training the PointFiveFour neural network on the Physionet dataset.")

    parser.add_argument('--batch_size', type=int, default=1, help='Input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train (default: 10)')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer (default: 1e-3)')
    parser.add_argument('--model_path', type=str, default='./model.pth', help='Path to save the trained model (default: ./model.pth)')
    parser.add_argument('--device', type=str, default='cuda', help='Use CUDA for training if available')

    return parser.parse_args()


def create_mask(labels):
    mask = labels != -1
    labels = labels[mask]
    return mask


def main():
    args = parse_arguments()

    # Define model
    # torch.manual_seed(42)
    # device = torch.device(config['summary_device'])
    # model = PointFiveFourModel(13)
    # model.to(device)

    # best_model = copy.deepcopy(model)
    # best_auprc = 0.0
    # best_epoch = 0

    # summary(model,
    #         eval(config['summary_shape']),
    #         device=device,
    #         verbose=1,
    #         depth=5,
    #         col_names=[
    #             'input_size',
    #             'output_size',
    #             "num_params",
    #             "params_percent",
    #             "kernel_size",
    #             "mult_adds",
    #             "trainable",
    #         ])

    # Set up DataLoaders
    train_dataset = PhysionetDataset(dir=train_dir,
                                     stride=dataloader_stride,
                                     order=filter_order,
                                     Wn=filter_Wn,
                                     pad_to_2_power_23_bool=pad_to_2_power_23)
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              collate_fn=collate_fn,
                              batch_size=batch_size,
                              num_workers=num_workers)

    val_dataset = PhysionetDataset(dir=val_dir,
                                   train=False,
                                   stride=dataloader_stride,
                                   order=filter_order,
                                   Wn=filter_Wn,
                                   pad_to_2_power_23_bool=pad_to_2_power_23)
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            collate_fn=collate_fn,
                            batch_size=batch_size,
                            num_workers=num_workers)

    test_dataset = PhysionetDataset(dir=test_dir,
                                    train=False,
                                    stride=dataloader_stride,
                                    order=filter_order,
                                    Wn=filter_Wn,
                                    pad_to_2_power_23_bool=pad_to_2_power_23)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             collate_fn=collate_fn,
                             batch_size=batch_size,
                             num_workers=num_workers)

    print('============ LOADER LENGTH ============')
    print('Train Loader length: ' + str(len(train_loader)))
    print('Val Loader length: ' + str(len(val_loader)))

    # Set up loss and optimizer
    device = torch.device('cuda')
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight])).to(device)
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                              max_lr=max_lr,
                                              pct_start=warmstart_percentage,
                                              steps_per_epoch=int(
                                                  len(train_loader)),
                                              epochs=epochs,
                                              anneal_strategy='linear')
    writer = SummaryWriter(log_dir=config['summary_writer'])
    total_steps = 0

    # Final sigmoid activation
    sigmoid = torch.nn.Sigmoid()

    # Training and Validation
    for epoch in range(1, epochs + 1):

        print('============ TRAIN EPOCH: ' + str(epoch) + ' ============')
        model.train()
        train_loss = 0
        for batch_idx, _data in enumerate(train_loader):
            # Send to CUDA
            inputs, labels = _data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Measure time during forward and backward propagation
            start = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            mask = create_mask(labels)
            loss = criterion(outputs[mask], labels[mask])

            loss.backward()
            optimizer.step()
            scheduler.step()
            end = time.time()

            total_steps += 1
            train_loss += loss

            writer.add_scalar('Train/Batch_Time', end - start, total_steps)
            writer.add_scalar('Train/LR',
                              scheduler.get_last_lr()[0], total_steps)

        writer.add_scalar('Train/Loss', train_loss / len(train_loader), epoch)
        writer.add_scalar('Train/Total_Steps', total_steps, epoch)
        writer.flush()

        if epoch == mem_snapshot_epochs:
            export_memory_snapshot(config['logging_folder'])
            stop_record_memory_history()

        print('============ VALIDATE EPOCH: ' + str(epoch) + ' ============')
        with torch.no_grad():
            # do not uncomment model.eval() due to both BCELoss and AUPRC calculation
            model.eval()
            val_loss = 0
            score = Challenge2018Score()
            for batch_idx, _data in enumerate(val_loader):
                # Send to CUDA
                inputs, labels = _data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Measure time during forward propagation
                start = time.time()
                outputs = model(inputs)
                end = time.time()

                # Calculate loss
                mask = create_mask(labels)
                loss = criterion(outputs[mask], labels[mask])
                val_loss += loss

                # Add final sigmoid activation
                outputs = sigmoid(outputs)

                # Send to CPU
                outputs = outputs.cpu().detach().numpy()
                labels = labels.cpu().detach().numpy()
                outputs = np.squeeze(outputs, axis=2)
                labels = np.squeeze(labels, axis=2)

                # Calculate AUPRC
                # TODO: parallelize
                for i, (output, label) in enumerate(zip(outputs, labels)):
                    record_name = str(batch_idx) + ' ' + str(i)
                    score.score_record(label, output, record_name)

                writer.add_scalar('Validation/Batch_Time', end - start,
                                  total_steps)

            auroc_g = score.gross_auroc()
            auprc_g = score.gross_auprc()
            print('Validation AUROC Performance (gross): %f' % auroc_g)
            print('Validation AUPRC Performance (gross): %f' % auprc_g)
            writer.add_scalar('Validation/AUROC', auroc_g, epoch)
            writer.add_scalar('Validation/AUPRC', auprc_g, epoch)
            writer.add_scalar('Validation/Loss',
                              val_loss / len(val_loader), epoch)

            # save best model
            if best_auprc < auprc_g:
                if os.path.exists(config['model_name'] + '_' + str(best_epoch)):
                    os.remove(config['model_name'] + '_' + str(best_epoch))

                best_auprc = auprc_g
                best_model = copy.deepcopy(model)
                best_epoch = epoch

                torch.save(best_model.state_dict(), config['model_name'] + '_' + str(best_epoch))

            writer.flush()
    writer.close()

    print('-------------------------- SAVE MODEL -------------------------')

    if os.path.exists(config['model_name']):
        os.remove(config['model_name'])
    print('best auprc: ' + str(best_auprc))
    print('best epoch: ' + str(best_epoch))
    torch.save(best_model.state_dict(), config['model_name'])

    print('============ TESTING ============')
    file = open(config['csv_file'], mode='w', newline='')
    writer = csv.writer(file)
    writer.writerow(["Name", "AUPRC", "AUROC"])

    with torch.no_grad():
        model.eval()
        score = Challenge2018Score()
        for batch_idx, _data in enumerate(test_loader):
            # Send to CUDA
            inputs, labels = _data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward propagation
            outputs = best_model(inputs, True)

            # Send to CPU
            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            outputs = np.squeeze(outputs, axis=2)
            labels = np.squeeze(labels, axis=2)

            # Calculate AUPRC
            # TODO: parallelize
            for i, (output, label) in enumerate(zip(outputs, labels)):
                record_name = str(batch_idx) + ' ' + str(i)
                score.score_record(label, output, record_name)
                curr_auprc = score.record_auprc(record_name)
                curr_auroc = score.record_auroc(record_name)
                writer.writerow([record_name, curr_auprc, curr_auroc])

        auroc_g = score.gross_auroc()
        auprc_g = score.gross_auprc()
        print('Testing AUROC Performance (gross): %f' % auroc_g)
        print('Testing AUPRC Performance (gross): %f' % auprc_g)

    file.close()
    print('Data has been written to the csv file.')

if __name__ == '__main__':
    main()
