import gc as garbageCollector
import os
import sys

import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary

from DeepSleep import DeepSleep
from DeepSleepDataset import DeepSleepDataset
from dummy_model import DummyModel
from MambaDeepSleep import MambaDeepSleep
from MambaSleepNet import MambaSleepNet
from SleepNet import Sleep_model_MultiTarget
from SleepNetDataset import SleepNetDataset
# from ssm import StateSpaceModel
from train import train_loop


def main():
    if len(sys.argv) == 2:
        settings_name = sys.argv[1].strip()
    else:
        print('How to use: python main.py settings_local.yaml')
        return

    torch.manual_seed(0)

    settings = OmegaConf.load(settings_name)
    splits = OmegaConf.load(OmegaConf.to_container(settings, resolve=True)['splits_yaml'])
    model = OmegaConf.load(OmegaConf.to_container(settings, resolve=True)['model_yaml'])
    settings = OmegaConf.merge(settings, splits, model)
    settings = OmegaConf.to_container(settings, resolve=True)
    print(settings)
    exit()

    if settings['model_name'] == 'DeepSleep':
        model = DeepSleep()
    elif settings['model_name'] == 'SleepNet':
        model = Sleep_model_MultiTarget(settings)
    elif settings['model_name'] == 'MambaSleepNet':
        model = MambaSleepNet(settings)
    elif settings['model_name'] == 'SleepNet':
        model = MambaDeepSleep()
    model.to(settings['device'])

    if model.name == 'DeepSleep':
        train_data = DeepSleepDataset('train', settings)
        validation_data = DeepSleepDataset('validation', settings)
        summary_shape = (1, 1048576, 13)
    elif model.name == 'SleepNet':
        train_data = SleepNetDataset('train', settings)
        validation_data = SleepNetDataset('validation', settings)
        summary_shape = (1, 1_260_000, 12)

    train_dataloader = DataLoader(train_data, batch_size=settings['train_batch_size'], shuffle=True, num_workers=settings['num_workers'])
    validation_dataloader = DataLoader(validation_data, batch_size=settings['test_batch_size'], shuffle=True, num_workers=settings['num_workers'])

    summary(model,
            summary_shape,
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
