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
                    settings, total_steps, epoch, writer):
    model.train()
    train_loss = 0.0

    for batch_idx, _data in enumerate(dataloader):
        inputs, labels = _data
        inputs = inputs.to(settings['device'])
        labels = labels.to(settings['device'])

        start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)

        mask = create_mask(labels)
        loss = criterion(outputs[mask], labels[mask])

        loss.backward()
        optimizer.step()
        scheduler.step()
        end = time.time()

        train_loss += loss
        total_steps += 1

        writer.add_scalar('train/batch_time', end - start, total_steps)
        writer.add_scalar('train/lr', scheduler.get_last_lr()[0], total_steps)

    writer.add_scalar('train/loss', train_loss / len(dataloader), epoch)
    writer.add_scalar('train/total_steps', total_steps, epoch)


def validate(model, dataloader, criterion, settings, total_steps, epoch,
             writer):
    model.eval()
    score = Challenge2018Score()
    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, _data in enumerate(dataloader):
            inputs, labels = _data
            inputs = inputs.to(settings['device'])
            labels = labels.to(settings['device'])

            start = time.time()
            outputs = model(inputs)

            mask = create_mask(labels)
            loss = criterion(outputs[mask], labels[mask])
            val_loss += loss

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            # print(outputs.shape)
            # print(labels.shape)
            # outputs = np.squeeze(outputs, axis=2)
            # labels = np.squeeze(labels, axis=2)

            for i, (output, label) in enumerate(zip(outputs, labels)):
                record_name = str(batch_idx) + ' ' + str(i)
                # print(record_name)
                # print(output.shape)
                # print(output)
                score.score_record(label, output, record_name)
            end = time.time()

            writer.add_scalar('validation/batch_time', end - start,
                              total_steps)

        auroc_g = score.gross_auroc()
        auprc_g = score.gross_auprc()
        print('AUPRC: %f, AUROC: %f' % (auprc_g, auroc_g))
        writer.add_scalar('validation/AUROC', auroc_g, epoch)
        writer.add_scalar('validation/AUPRC', auprc_g, epoch)
        writer.add_scalar('validation/loss', val_loss / len(dataloader), epoch)

    return auprc_g


def train_loop(model, train_dataloader, validation_dataloader, settings):
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([settings['pos_weight']])).to(
            settings['device'])
    optimizer = optim.AdamW(model.parameters())
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(settings['max_lr']),
        pct_start=settings['warmstart_percentage'],
        steps_per_epoch=int(len(train_dataloader)),
        epochs=settings['epochs'],
        anneal_strategy='linear')

    total_steps = 0
    best_auprc = 0.0
    best_epoch = 0
    # best_model = copy.deepcopy(model)
    writer = SummaryWriter(log_dir=settings['summary_writer'])

    for epoch in range(1, settings['epochs'] + 1):
        print('============ EPOCH: ' + str(epoch) + ' ============',
              flush=True)
        train_one_epoch(model, train_dataloader, criterion, scheduler,
                        optimizer, settings, total_steps, epoch, writer)
        val_auprc = validate(model, validation_dataloader, criterion, settings,
                             total_steps, epoch, writer)

        if best_auprc < val_auprc:
            print('best auprc: %f, best epoch: %d' % (val_auprc, epoch))
            if os.path.exists(settings['model_name']):
                os.remove(settings['model_name'])

            best_auprc = val_auprc
            # best_model = copy.deepcopy(model)
            best_epoch = epoch

            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, settings['model_name'])

        writer.flush()
    writer.close()
