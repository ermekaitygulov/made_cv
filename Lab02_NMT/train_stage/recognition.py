import math
from collections import deque
import os

import editdistance
import numpy as np
import torch
import wandb
from torch import optim
from torch.nn.functional import ctc_loss, log_softmax
from tqdm import tqdm

from utils import ALPHABET


class RecognitionStage:
    default_config = {
        'opt_class': 'Adam',
        'opt_params': {},
        'log_window_size': 10,
        'epoch': 10,
        'save_mod': 5,
    }

    def __init__(self, model, stage_name, device, stage_config):
        self.config = self.default_config.copy()
        self.config.update(stage_config)
        self.name = stage_name
        self.model = model
        self.opt = self.init_opt()
        self.lr_scheduler = self.init_scheduler()
        self.criterion = ctc_loss
        self.device = device

    def train(self, train_iterator, val_iterator):
        train_step = 0
        best_val_acc = -1
        for epoch in range(self.config['epoch']):
            train_loss, train_step = self.train_epoch(
                train_iterator,
                train_step
            )

            val_acc, val_acc_ed = self.val_epoch(val_iterator, epoch)

            if (epoch + 1) % self.config['save_mod'] == 0:
                self.save_model(f'{self.name}-rec_model{epoch}.pt')
            if val_acc > best_val_acc:
                self.save_model(f'{self.name}-rec_model_best.pt')
                best_val_acc = val_acc

            print(f'Epoch: {epoch + 1:02}')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. acc: {val_acc:.3f} |  Val. acc_ed: {math.exp(val_acc_ed):7.3f}'
                  f' (best: {best_val_acc:.3f})')
        self.save_model(f'{self.name}-rec_model_last.pt')

    def train_epoch(self, iterator, global_step):
        self.model.train()
        epoch_loss = 0
        loss_window = deque(maxlen=self.config['log_window_size'])
        tqdm_iterator = tqdm(iterator)

        for i, batch in enumerate(tqdm_iterator):
            self.opt.zero_grad()

            images = batch["images"].to(self.device)
            seqs = batch["seqs"]
            seq_lens = batch["seq_lens"]

            # TODO TIP: What happens here is explained in seminar 06.
            seqs_pred = self.model(images).cpu()
            log_probs = log_softmax(seqs_pred, dim=2)
            seq_lens_pred = torch.Tensor([seqs_pred.size(0)] * seqs_pred.size(1)).int()

            loss = self.criterion(log_probs, seqs, seq_lens_pred, seq_lens)
            epoch_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.opt.step()
            if self.lr_scheduler:
                self.lr_scheduler.step()
            loss_window.append(loss.cpu().detach().numpy())
            if (i + 1) % self.config['log_window_size'] == 0:
                log_dict = dict()
                mean_loss = np.mean(loss_window)
                log_dict['train_loss'] = mean_loss
                log_dict['train_step'] = global_step
                log_dict['learning_rate'] = self.opt.param_groups[0]["lr"]

                if tqdm_iterator._ema_dt():
                    log_dict['train_speed(batch/sec)'] = tqdm_iterator._ema_dn() / tqdm_iterator._ema_dt()
                if wandb.run:
                    wandb.log({self.name: log_dict})
                tqdm_iterator.set_postfix(train_loss=mean_loss)

        global_step += 1

        return epoch_loss / len(iterator), global_step

    def val_epoch(self, iterator, epoch):
        self.model.eval()
        count, tp, avg_ed = 0, 0, 0
        tqdm_iterator = tqdm(iterator)

        with torch.no_grad():
            for batch in tqdm_iterator:
                images = batch["images"].to(self.device)
                out = self.model(images, decode=True)

                gt = (batch["seqs"].numpy() - 1).tolist()
                lens = batch["seq_lens"].numpy().tolist()

                pos, key = 0, ''
                for i in range(len(out)):
                    gts = ''.join(ALPHABET[c] for c in gt[pos:pos + lens[i]])
                    pos += lens[i]
                    if gts == out[i]:
                        tp += 1
                    else:
                        avg_ed += editdistance.eval(out[i], gts)
                    count += 1

        acc = tp / count
        avg_ed = avg_ed / count
        if wandb.run:
            wandb.log({'Val acc': acc, 'Val acc ed': avg_ed, 'epoch': epoch})
        return acc, avg_ed

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
