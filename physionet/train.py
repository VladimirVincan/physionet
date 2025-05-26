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


def sleep_net_loss(predictions, truths, settings):
    arousal_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=True).to(settings['device'])
    apnea_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=True).to(settings['device'])
    wake_criterion = torch.nn.CrossEntropyLoss(ignore_index=-1, reduce=True).to(settings['device'])

    arousal_outputs, apnea_hypopnea_outputs, sleep_stage_outputs = predictions
    batch_arousal_targs, batch_apnea_targs, batch_wake_targs = truths

    batch_arousal_targs = batch_arousal_targs.view(-1)
    batch_apnea_targs = batch_apnea_targs.view(-1)
    batch_wake_targs = batch_wake_targs.view(-1)

    arousal_outputs = arousal_outputs.permute(0, 2, 1).contiguous().view(-1, 2)
    apnea_hypopnea_outputs = apnea_hypopnea_outputs.permute(0, 2, 1).contiguous().view(-1, 2)
    sleep_stage_outputs = sleep_stage_outputs.permute(0, 2, 1).contiguous().view(-1, 2)

    arousal_loss = arousal_criterion(arousal_outputs, batch_arousal_targs)
    apnea_hypopnea_loss = apnea_criterion(apnea_hypopnea_outputs, batch_apnea_targs)
    sleep_stage_loss = wake_criterion(sleep_stage_outputs, batch_wake_targs)

    loss = ((2*arousal_loss) + apnea_hypopnea_loss + sleep_stage_loss) / 4.0

    return loss


def train_one_epoch(model, dataloader, criterion, scheduler, optimizer,
                    settings, current_params, writer):
    model.train()
    train_loss = 0.0
    epoch_start = time.time()

    for batch_idx, _data in enumerate(dataloader):
        inputs, labels = _data  # TODO: add num_samples
        inputs = inputs.to(settings['device'])
        if model.name != 'SleepNet':
            labels = labels.to(settings['device'])

        batch_start = time.time()
        optimizer.zero_grad()
        outputs = model(inputs)

        if model.name == 'SleepNet':
            loss = criterion(outputs, labels, settings)
        else:
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

        if model.name == 'SleepNet':
            if batch_idx == 100:
                break

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
            if model.name != 'SleepNet':
                labels = labels.to(settings['device'])

            start = time.time()
            outputs = model(inputs)

            if model.name == 'SleepNet':
                loss = criterion(outputs, labels, settings)
                labels = labels[0]
                print(labels.shape)
                outputs = outputs[0][:, 1]
                print(outputs.shape)
            else:
                mask = create_mask(labels)
                loss = criterion(outputs[mask], labels[mask])
            val_loss += loss

            if criterion == nn.BCEWithLogitsLoss:
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
    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=torch.tensor([settings['pos_weight']])).to(
    #         settings['device'])
    criterion = sleep_net_loss
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
