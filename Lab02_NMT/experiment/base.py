import time
from abc import ABC, abstractmethod
from typing import Type, Dict

import torch

from utils import Task

ABC_NN_CATALOG = {}


class Experiment(ABC):
    nn_catalog = ABC_NN_CATALOG

    def __init__(self, config, device):
        self.config = config
        self.device = device

        self.task = Task(*self.read_data())
        self.model = self.init_model()
        self.trainer = self.init_trainer()
        self.time = None

    def train(self):
        start_time = time.time()
        train_iterator, val_iterator = self.task
        self.trainer.train(train_iterator, val_iterator)
        end_time = time.time()
        self.time = end_time - start_time

    def init_model(self):
        model_config = self.config['model']
        model_class = self.nn_catalog[model_config['name']]
        model = model_class(**model_config['params'])

        if 'model_path' in self.config:
            with open(self.config['model_path'], "rb") as fp:
                state_dict = torch.load(fp, map_location='cpu')
                model.load_state_dict(state_dict)
        model.to(self.device)
        return model

    @abstractmethod
    def init_trainer(self):
        raise NotImplementedError

    @abstractmethod
    def read_data(self):
        raise NotImplementedError


EXPERIMENT_CATALOG: Dict[str, Type[Experiment]] = {}
