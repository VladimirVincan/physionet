"""
Inputs:
  - model_name
  - settings.yaml
Outputs:
  - test_results.csv: for each file in test, print auprc, aoroc, loss
"""

import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from dataset import (DeepSleepDataset, NormalizedPhysionetDataset,
                     PhysionetDataset)
from dummy_model import DummyModel
from score2018 import Challenge2018Score
from utils import create_mask


def test(model, dataloader, criterion, settings):
    model.eval()
    score = Challenge2018Score()
    test_loss = 0.0

    data = []
    data.append(['file name', 'auprc', 'auroc', 'loss'])

    with torch.no_grad():
        for batch_idx, _data in enumerate(dataloader):
            inputs, labels, file_name = _data
            inputs = inputs.to(settings['device'])
            labels = labels.to(settings['device'])

            outputs = model(inputs)

            mask = create_mask(labels)
            loss = criterion(outputs[mask], labels[mask])
            test_loss += loss

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            # TODO: remove for loop, single element in array
            for i, (output, label) in enumerate(zip(outputs, labels)):
                score.score_record(label, output, file_name)
                data.append([file_name, score.record_auprc(file_name), score.record_auroc(file_name), loss])

    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()

    with open(settings['test_csv_name'], mode='w', newline='\n') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def main():
    if len(sys.argv) == 2:
        settingsName = sys.argv[1].strip()
    else:
        print('How to use: python main.py settings_local.yaml')
        return

    with open(settingsName, 'r') as file:
        settings = yaml.safe_load(file)

    model = DummyModel()
    model.load_state_dict(torch.load(settings['model_name'])['model_state_dict'])
    test_data = PhysionetDataset('test', settings)
    test_dataloader = DataLoader(test_data, batch_size=settings['test_batch_size'], shuffle=False, num_workers=settings['num_workers'])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([settings['pos_weight']])).to(
            settings['device'])

    test(model, test_dataloader, criterion, settings)


if __name__ == '__main__':
    main()
