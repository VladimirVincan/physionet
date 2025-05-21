# import copy
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from score2018 import Challenge2018Score
from utils import create_mask


def train_one_epoch(model, dataloader, criterion, scheduler, optimizer,
                    settings, current_params, writer):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch_idx, _data in enumerate(dataloader):
        inputs, labels = _data  # TODO: add num_samples
        inputs = inputs.to(settings['device'])
        labels = labels.to(settings['device'])

        batch_start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)

        mask = create_mask(labels)
        loss = criterion(outputs[mask], labels[mask])

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        batch_end = time.time()

        train_loss += loss
        current_params['total_steps'] += 1

        writer.add_scalar('train/batch_time', batch_end - batch_start, current_params['total_steps'])
        if scheduler is not None:
            writer.add_scalar('train/lr', scheduler.get_last_lr()[0], current_params['total_steps'])
        else:
            writer.add_scalar('train/lr', float(settings['max_lr']), current_params['total_steps'])

    epoch_end = time.time()
    writer.add_scalar('train/epoch_time', epoch_end - epoch_start, current_params['epoch'])
    writer.add_scalar('train/loss', train_loss / len(dataloader), current_params['epoch'])
    writer.add_scalar('train/total_steps', current_params['total_steps'], current_params['epoch'])


def validate(model, dataloader, criterion, settings, current_params,
             writer):
    model.eval()
    score = Challenge2018Score()
    val_loss = 0.0
    epoch_start = time.time()

    with torch.no_grad():
        for batch_idx, _data in enumerate(dataloader):
            inputs, labels, _, num_samples = _data
            inputs = inputs.to(settings['device'])
            labels = labels.to(settings['device'])

            start = time.time()
            outputs = model(inputs)

            mask = create_mask(labels)
            loss = criterion(outputs[mask], labels[mask])
            val_loss += loss

            sigmoid = nn.Sigmoid()
            outputs = sigmoid(outputs)  # use with BCEWithLogitsLoss

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            outputs = outputs[:, :num_samples]
            labels = labels[:, :num_samples]

            for i, (output, label) in enumerate(zip(outputs, labels)):
                record_name = str(batch_idx) + ' ' + str(i)
                # print(record_name)
                # print(output.shape)
                # print(output)
                score.score_record(label, output, record_name)
            end = time.time()

            current_params['total_steps'] += 1

            writer.add_scalar('validation/batch_time', end - start,
                              current_params['total_steps'])

        auroc_g = score.gross_auroc()
        auprc_g = score.gross_auprc()
        epoch_end = time.time()

        print('AUPRC: %f, AUROC: %f' % (auprc_g, auroc_g))
        writer.add_scalar('validation/AUROC', auroc_g, current_params['epoch'])
        writer.add_scalar('validation/AUPRC', auprc_g, current_params['epoch'])
        writer.add_scalar('validation/loss', val_loss / len(dataloader), current_params['epoch'])
        writer.add_scalar('validation/epoch_time', epoch_end - epoch_start, current_params['epoch'])

    return auprc_g


def train_loop(model, train_dataloader, validation_dataloader, settings):
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([settings['pos_weight']])).to(
            settings['device'])
    # criterion = nn.BCELoss().to(settings['device'])
    optimizer = optim.AdamW(model.parameters(), lr=float(settings['max_lr']), weight_decay=float(settings['weight_decay']), betas=(settings['beta_1'], settings['beta_2']))
    # scheduler = optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=float(settings['max_lr']),
    #     pct_start=settings['warmstart_percentage'],
    #     steps_per_epoch=int(len(train_dataloader)),
    #     epochs=settings['epochs'],
    #     anneal_strategy='linear')
    scheduler = None

    current_params = {
        'total_steps': 0,
        'epoch': 0,
        'best_auprc': 0.0,
        'best_epoch': 0
    }

    # best_model = copy.deepcopy(model)
    writer = SummaryWriter(log_dir=settings['summary_writer'])

    for epoch in range(1, settings['epochs'] + 1):
        print('============ EPOCH: ' + str(epoch) + ' ============',
              flush=True)
        train_one_epoch(model, train_dataloader, criterion, scheduler,
                        optimizer, settings, current_params, writer)
        val_auprc = validate(model, validation_dataloader, criterion, settings,
                             current_params, writer)
        current_params['epoch'] = epoch

        if current_params['best_auprc'] < val_auprc:
            print('best auprc: %f, best epoch: %d' % (val_auprc, epoch))
            if os.path.exists(settings['model_name']):
                os.remove(settings['model_name'])

            current_params['best_auprc'] = val_auprc
            # best_model = copy.deepcopy(model)
            current_params['best_epoch'] = epoch

            torch.save(
                {
                    'current_params': current_params,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, settings['model_name'])

        writer.flush()
    writer.close()
