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
from DeepSleep import DeepSleep
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
            inputs, labels, filepaths_dict, num_samples = _data
            folder_name = filepaths_dict['folder_name'][0]
            outputs_path = filepaths_dict['outputs_path'][0]

            inputs = inputs.to(settings['device'])
            labels = labels.to(settings['device'])

            outputs = model(inputs)

            mask = create_mask(labels)
            loss = criterion(outputs[mask], labels[mask])
            test_loss += loss

            outputs = outputs.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()

            outputs = outputs[:, :num_samples]
            labels = labels[:, :num_samples]

            np.save(outputs_path, outputs)
            print('saving: ' + outputs_path)

            # TODO: remove for loop, single element in array
            for i, (output, label) in enumerate(zip(outputs, labels)):
                score.score_record(label, output, folder_name)
                data.append([folder_name, score.record_auprc(folder_name), score.record_auroc(folder_name), loss])

            print(folder_name + ' | auprc: ' + str(score.record_auprc(folder_name)) + ' auroc: ' + str(score.record_auroc(folder_name)))

    auroc_g = score.gross_auroc()
    auprc_g = score.gross_auprc()

    print('total auprc: ' + str(auprc_g) + ' total auroc: ' + str(auroc_g))

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

    model = DeepSleep()
    test_data = DeepSleepDataset(settings['split'], settings)
    # model = DummyModel()
    # test_data = PhysionetDataset('test', settings)

    model.load_state_dict(torch.load(settings['model_name'], weights_only=False)['model_state_dict'])
    model.to(settings['device'])

    test_dataloader = DataLoader(test_data, batch_size=settings['test_batch_size'], shuffle=False, num_workers=settings['num_workers'])
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([settings['pos_weight']])).to(
            settings['device'])

    test(model, test_dataloader, criterion, settings)


if __name__ == '__main__':
    main()
