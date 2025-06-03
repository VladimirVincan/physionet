import gc as garbageCollector
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from torchinfo import summary

from dataset import (DeepSleepDataset, NormalizedPhysionetDataset,
                     PhysionetDataset, SleepNetDataset)
from DeepSleep import DeepSleep
from dummy_model import DummyModel
from MambaDeepSleep import MambaDeepSleep
from MambaSleepNet import MambaSleepNet
from SleepNet import Sleep_model_MultiTarget
# from ssm import StateSpaceModel
from train import train_loop


def main():
    if len(sys.argv) == 2:
        settings_name = sys.argv[1].strip()
    else:
        print('How to use: python main.py settings_local.yaml')
        return

    torch.manual_seed(0)

    with open(settings_name, 'r') as file:
        settings = yaml.safe_load(file)

    dataloader_stride = 1
    # model = StateSpaceModel("[(16, 8, 8)] + [(64, 5, 5)]", "3*[(64)]", "[(64, 1, 1)]", dataloader_stride)
    # model = DummyModel("[(16, 8, 8)] + [(64, 5, 5)]", "3*[(64)]", "[(64, 1, 1)]", dataloader_stride)
    model = DeepSleep()
    # model = Sleep_model_MultiTarget(settings)
    # model = MambaSleepNet(settings)
    # model = MambaDeepSleep()
    model.to(settings['device'])

    # train_data = NormalizedPhysionetDataset('train', settings)
    # validation_data = NormalizedPhysionetDataset('validation', settings)
    # test_data = PhysionetDataset('test', settings)

    if model.name == 'DeepSleep':
        train_data = DeepSleepDataset('train', settings)
        validation_data = DeepSleepDataset('validation', settings)
    elif model.name == 'SleepNet':
        train_data = SleepNetDataset('train', settings)
        validation_data = SleepNetDataset('validation', settings)

    train_dataloader = DataLoader(train_data, batch_size=settings['train_batch_size'], shuffle=True, num_workers=settings['num_workers'])
    validation_dataloader = DataLoader(validation_data, batch_size=settings['test_batch_size'], shuffle=True, num_workers=settings['num_workers'])


    summary(model,
            eval(settings['summary_shape']),
            device=settings['device'],
            verbose=1,
            depth=5,
            col_names=[
                'input_size',
                'output_size',
                "num_params",
                "params_percent",
                "kernel_size",
                "mult_adds",
                "trainable",
            ])

    train_loop(model, train_dataloader, validation_dataloader, settings)

if __name__ == '__main__':
    main()
