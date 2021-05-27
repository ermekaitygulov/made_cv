import os
from abc import ABC, abstractmethod

import torch
import wandb
from torch import optim


class BaseStage(ABC):
    default_config = {
    }

    def __init__(self, model, stage_name, device, stage_config):
        self.config = self.default_config.copy()
        self.config.update(stage_config)
        self.name = stage_name
        self.model = model
        self.opt = self.init_opt()
        self.lr_scheduler = self.init_scheduler()
        self.device = device

    @abstractmethod
    def train(self, train_iterator, val_iterator):
        raise NotImplementedError

    def save_model(self, name):
        if wandb.run:
            save_path = os.path.join('model_save', wandb.run.name)
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.model.state_dict(), os.path.join(save_path, name))
        else:
            torch.save(self.model.state_dict(), name)

    def init_opt(self):
        opt_class = getattr(optim, self.config['opt_class'])
        opt_params = self.config['opt_params']
        opt = opt_class(self.model.parameters(), **opt_params)
        return opt

    def init_scheduler(self):
        # TODO refactor
        if 'scheduler_class' not in self.config:
            return None
        scheduler_class = getattr(optim.lr_scheduler, self.config['scheduler_class'])
        scheduler_params = self.config['scheduler_params']
        scheduler = scheduler_class(self.opt, **scheduler_params)
        return scheduler
