from collections import namedtuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

Task = namedtuple('Task', ['train', 'val'])


def add_to_catalog(name, catalog):
    def add_wrapper(class_to_add):
        catalog[name] = class_to_add
        return class_to_add
    return add_wrapper


def pred_to_string(pred, abc):
    seq = []
    for i in range(len(pred)):
        label = np.argmax(pred[i])
        seq.append(label - 1)
    out = []
    for i in range(len(seq)):
        if len(out) == 0:
            if seq[i] != -1:
                out.append(seq[i])
        else:
            if seq[i] != -1 and seq[i] != seq[i - 1]:
                out.append(seq[i])
    out = ''.join([abc[c] for c in out])
    return out


def decode_sequence(pred, abc):
    pred = pred.permute(1, 0, 2).cpu().data.numpy()
    outputs = []
    for i in range(len(pred)):
        outputs.append(pred_to_string(pred[i], abc))
    return outputs


ALPHABET = "0123456789ABCEHKMOPTXY"
MAPPING = {
    'А': 'A',
    'В': 'B',
    'С': 'C',
    'Е': 'E',
    'Н': 'H',
    'К': 'K',
    'М': 'M',
    'О': 'O',
    'Р': 'P',
    'Т': 'T',
    'Х': 'X',
    'У': 'Y',
}


def labels_to_text(labels, abc=ALPHABET):
    return ''.join(list(map(lambda x: abc[int(x) - 1], labels)))


def text_to_labels(text, abc=ALPHABET):
    return list(map(lambda x: abc.index(x) + 1, text))


def is_valid_str(s, abc=ALPHABET):
    for ch in s:
        if ch not in abc:
            return False
    return True


def convert_to_eng(text, mapping=None):
    if mapping is None:
        mapping = MAPPING
    return ''.join([mapping.get(a, a) for a in text])


def dice_coeff(input, target):
    smooth = 1.

    input_flat = input.view(-1)
    target_flat = target.view(-1)
    intersection = (input_flat * target_flat).sum()
    union = input_flat.sum() + target_flat.sum()

    return (2. * intersection + smooth) / (union + smooth)


def dice_loss(input, target):
    # TODO TIP: Optimizing the Dice Loss usually helps segmentation a lot.
    return - torch.log(dice_coeff(input, target))


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            bce_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt)**self.gamma * bce_loss

        if self.reduce:
            return torch.mean(focal_loss)
        else:
            return focal_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # True Positives, False Positives & False Negatives
        true_positive = (inputs * targets).sum()
        false_positive = ((1 - targets) * inputs).sum()
        false_negative = (targets * (1 - inputs)).sum()

        tversky = (true_positive + 1.) / (true_positive + self.alpha * false_positive + self.beta * false_negative + 1.)

        return - torch.log(tversky)