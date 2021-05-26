import os

from torch.utils.data import DataLoader

from dataset.recognition import RecognitionDataset
from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from neural_network import REC_NN_CATALOG
from train_stage import RecognitionStage
from transforms import get_train_transforms, get_val_transforms
from utils import add_to_catalog, ALPHABET


@add_to_catalog('recognition', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    nn_catalog = REC_NN_CATALOG

    def init_trainer(self):
        stage = RecognitionStage(self.model, 'Adam_stage', self.device, self.config['Adam_stage'])
        return stage

    def read_data(self):
        data_config = self.config['data']
        train_transforms = get_train_transforms(**data_config['train_transforms'])

        data_path = data_config['path']
        train_dataset = RecognitionDataset(data_path, os.path.join(data_path, "train_recognition.json"),
                                           abc=ALPHABET, transforms=train_transforms, split="train")
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=data_config['batch_size'],
                                      shuffle=True,
                                      num_workers=data_config['num_workers'],
                                      collate_fn=train_dataset.collate_fn)

        val_transforms = get_val_transforms(**data_config['val_transforms'])
        val_dataset = RecognitionDataset(data_path, os.path.join(data_path, "train_recognition.json"),
                                         abc=ALPHABET, transforms=val_transforms, split="val")
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=data_config['batch_size'],
                                    shuffle=False,
                                    num_workers=data_config['num_workers'],
                                    collate_fn=val_dataset.collate_fn)

        return train_dataloader, val_dataloader
