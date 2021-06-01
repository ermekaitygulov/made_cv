from collections import deque

import numpy as np
import torch
from tqdm import tqdm
import wandb

from train_stage.base import BaseStage
from utils import dice_loss, dice_coeff, FocalLoss


class SegmentationStage(BaseStage):
    default_config = {
        'opt_class': 'Adam',
        'opt_params': {},
        'log_window_size': 10,
        'focal_gamma': 2,
        'epoch': 10,
        'save_mod': 5,
        'weight_focal': 1
    }

    def __init__(self, model, stage_name, device, stage_config):
        super(SegmentationStage, self).__init__(model, stage_name, device, stage_config)
        self.focal_loss = FocalLoss(gamma=self.config['focal_gamma'])
        self.dice_loss = dice_loss
        self.weight_focal = self.config['weight_focal']

    def train(self, train_iterator, val_iterator):
        train_step = 0
        best_val_dice = 0
        for epoch in range(self.config['epoch']):
            train_loss, train_focal, train_dice, train_step = self.train_epoch(
                train_iterator,
                train_step
            )

            val_dice = self.val_epoch(val_iterator, epoch)
            if self.lr_scheduler:
                self.lr_scheduler.step()
            if (epoch + 1) % self.config['save_mod'] == 0:
                self.save_model(f'{self.name}-seg_model{epoch}.pt')
            if val_dice > best_val_dice:
                self.save_model(f'{self.name}-seg_model_best.pt')
                best_val_dice = val_dice

            print(f'Epoch: {epoch + 1:02}')
            print(f'\tTrain Loss: {train_loss:.3f} |  focal: {train_focal:.3f} |  Dice: {train_dice:.3f}')
            print(f'\t Val. dice: {val_dice:.3f} (best: {best_val_dice:.3f})')
        self.save_model(f'{self.name}-seg_model_last.pt')

    def train_epoch(self, iterator, global_step):
        self.model.train()

        epoch_loss = 0
        epoch_focal_loss, epoch_dice_loss = 0, 0
        tqdm_iterator = tqdm(iterator)
        loss_window = deque(maxlen=self.config['log_window_size'])
        focal_window = deque(maxlen=self.config['log_window_size'])
        dice_window = deque(maxlen=self.config['log_window_size'])

        for i, batch in enumerate(tqdm_iterator):
            self.opt.zero_grad()
            imgs, true_masks = batch
            masks_pred = self.model(imgs.to(self.device))
            masks_probs = torch.sigmoid(masks_pred)

            loss_dict = self.criterion(masks_probs.cpu().view(-1), true_masks.view(-1))
            loss = loss_dict['total']
            focal = loss_dict['focal']
            dice = loss_dict['dice']

            epoch_focal_loss += focal.item()
            epoch_dice_loss += dice.item()
            epoch_loss += loss.item()
            loss_window.append(loss.item())
            focal_window.append(focal.item())
            dice_window.append(dice.item())

            loss.backward()
            self.opt.step()

            if(i + 1) % self.config['log_window_size'] == 0:
                log_dict = dict()
                mean_loss = np.mean(loss_window)
                mean_focal = np.mean(focal_window)
                mean_dice = np.mean(dice_window)
                log_dict['train_loss'] = mean_loss
                log_dict['train_focal'] = mean_focal
                log_dict['train_dice'] = mean_dice
                log_dict['train_step'] = global_step
                log_dict['learning_rate'] = self.opt.param_groups[0]["lr"]

                if tqdm_iterator._ema_dt():
                    log_dict['train_speed(batch/sec)'] = tqdm_iterator._ema_dn() / tqdm_iterator._ema_dt()
                if wandb.run:
                    wandb.log({self.name: log_dict})
                tqdm_iterator.set_postfix(train_loss=mean_loss, train_focal=mean_focal, train_dice=mean_dice)

            global_step += 1

        epoch_loss_mean = epoch_loss / len(iterator)
        epoch_focal_mean = epoch_focal_loss / len(iterator)
        epoch_dice_mean = epoch_dice_loss / len(iterator)
        return epoch_loss_mean, epoch_focal_mean, epoch_dice_mean, global_step

    def val_epoch(self, iterator, epoch):
        self.model.eval().to(self.device)

        val_dice = 0
        tqdm_iterator = tqdm(iterator)
        with torch.no_grad():
            for batch in tqdm_iterator:
                images, true_masks = batch
                masks_pred = self.model(images.to(self.device)).squeeze(1)  # (b, 1, h, w) -> (b, h, w)
                masks_pred = (torch.sigmoid(masks_pred) > 0.5).float()
                dice = dice_coeff(masks_pred.cpu(), true_masks).item()
                val_dice += dice

        val_dice /= len(iterator)
        if wandb.run:
            wandb.log({'Val dice': val_dice, 'epoch': epoch})
        return val_dice

    def criterion(self, x, y):
        focal = self.focal_loss(x, y)
        dice = self.dice_loss(x, y)
        total_loss = self.weight_focal * focal + (1. - self.weight_focal) * dice
        loss_dict = {
            'focal': focal,
            'dice': dice,
            'total': total_loss
        }
        return loss_dict