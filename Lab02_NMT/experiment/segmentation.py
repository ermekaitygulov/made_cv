import os

from torch import nn
from torch.utils.data import DataLoader

from dataset.segmentation import DetectionDataset
from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage import SegmentationStage
from neural_network import SEG_NN_CATALOG
from dataset.seg_transforms import get_train_transforms, get_val_transforms
from utils import add_to_catalog


@add_to_catalog('segmentation', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    nn_catalog = SEG_NN_CATALOG

    def __init__(self, config, device):
        super(Baseline, self).__init__(config, device)
        if config['model']['freeze_bn']:
            for layer in self.model.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.requires_grad_(False)

    def init_trainer(self):
        stage = SegmentationStage(self.model, 'Adam_stage', self.device, self.config['Adam_stage'])
        return stage

    def read_data(self):
        data_config = self.config['data']
        data_path = data_config['path']

        train_transforms = get_train_transforms(**data_config['train_transforms'])
        train_dataset = DetectionDataset(data_path, os.path.join(data_path, "train_segmentation.json"),
                                         transforms=train_transforms, split="train")
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=data_config['batch_size'],
                                      shuffle=True,
                                      num_workers=data_config['num_workers'],
                                      pin_memory=True,
                                      drop_last=True)

        val_transforms = get_val_transforms(**data_config['val_transforms'])
        val_dataset = DetectionDataset(data_path, os.path.join(data_path, "train_segmentation.json"),
                                       transforms=val_transforms, split="val")
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=data_config['batch_size'],
                                    shuffle=False,
                                    num_workers=data_config['num_workers'],
                                    pin_memory=True,
                                    drop_last=False)

        return train_dataloader, val_dataloader
