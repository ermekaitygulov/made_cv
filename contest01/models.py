import torchvision.models as models
import torch.nn as nn
import torch
import timm

from transforms import CROP_SIZE

CATALOG = {}


def add_to_catalog(name):
    def add_wrapper(class_to_add):
        CATALOG[name] = class_to_add
        return class_to_add
    return add_wrapper


@add_to_catalog('resnet')
class AvgResNet(nn.Module):
    def __init__(self, output_size):
        super(AvgResNet, self).__init__()
        self.model = timm.create_model('resnet34d', pretrained=True)
        # self.model.requires_grad_(False)

        fc_input = 512 * (CROP_SIZE // 32) ** 2
        x3_input = 256 * 4 ** 2
        x2_input = 128 * 8 ** 2
        self.model.fc = nn.Linear(fc_input + x3_input + x2_input, output_size, bias=True)
        self.x3_avg = nn.AdaptiveAvgPool2d((4, 4))
        self.x2_avg = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        x4 = torch.flatten(x4, 1)
        x3 = self.x3_avg(x3)
        x3 = torch.flatten(x3, 1)
        x2 = self.x2_avg(x2)
        x2 = torch.flatten(x2, 1)

        out = torch.cat([x4, x3, x2], 1)
        out = self.model.fc(out)
        return out


@add_to_catalog('efficientnet')
class EfficientNet(nn.Module):
    def __init__(self, output_size):
        super(EfficientNet, self).__init__()
        self.model = timm.create_model('efficientnet_b0', pretrained=True)
        self.model.requires_grad_(False)

        fc_input = 1280 * (CROP_SIZE // 32) ** 2

        self.fc = nn.Linear(fc_input, output_size, bias=True)
        self.model.conv_head.requires_grad_(True)
        self.model.blocks[-4:].requires_grad_(True)

    def forward(self, x):
        x = self.model.forward_features(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        return out


@add_to_catalog('resnetd')
def resnet_d(output_size):
    model = timm.create_model('resnet34d', pretrained=True)
    model.fc = nn.Linear(512, output_size, bias=True)
    return model