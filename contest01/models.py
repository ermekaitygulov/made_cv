import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import ops

CATALOG = {}


def add_to_catalog(name):
    def add_wrapper(class_to_add):
        CATALOG[name] = class_to_add
        return class_to_add
    return add_wrapper


@add_to_catalog('resnet')
class MyResNet(nn.Module):
    def __init__(self, output_size):
        super(MyResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.requires_grad_(False)

        fc_input = 512 * 4 ** 2
        s1_input = 256 * 4 ** 2
        s2_input = 128 * 8 ** 2
        self.model.fc = nn.Linear(fc_input + s1_input + s2_input, output_size, bias=True)
        self.model.fc.requires_grad_(True)
        self.model.layer4.requires_grad_(True)
        self.model.layer3.requires_grad_(True)
        self.model.layer2.requires_grad_(True)
        self.avg_pool1 = nn.AdaptiveAvgPool2d((4, 4))
        self.avg_pool2 = nn.AdaptiveAvgPool2d((8, 8))

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        x4 = torch.flatten(x4, 1)
        x3 = self.avg_pool1(x3)
        x3 = torch.flatten(x3, 1)
        x2 = self.avg_pool2(x2)
        x2 = torch.flatten(x2, 1)
        out = torch.cat([x4, x3, x2], 1)
        out = self.model.fc(out)
        return out


@add_to_catalog('resnetv2')
class MyResNetV2(nn.Module):
    def __init__(self, output_size):
        super(MyResNetV2, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.requires_grad_(False)

        fc_input = 512 * 4 ** 2
        s1_input = 256 * 8 ** 2
        self.model.fc = nn.Linear(fc_input + s1_input, output_size, bias=True)
        self.model.fc.requires_grad_(True)
        self.model.layer4.requires_grad_(True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x1 = self.model.layer1(x)
        x2 = self.model.layer2(x1)
        x3 = self.model.layer3(x2)
        x4 = self.model.layer4(x3)

        x4 = torch.flatten(x4, 1)
        x3 = torch.flatten(x3, 1)
        out = torch.cat([x4, x3], 1)
        out = self.model.fc(out)
        return out


class PyramidFeatures(nn.Module):

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        """
        FPN constructor.

        Args:
            - C3_size: num features in C3 map.
            - C4_size: num features in C4 map.
            - C5_size: num features in C5 map.
            - feature_size: num features in output maps.
        """
        super(PyramidFeatures, self).__init__()

        # For P5
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=(1, 1))
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=(3, 3), padding=(1, 1))

        # For P4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=(1, 1))
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=(3, 3), padding=(1, 1))

        # For P3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=(1, 1))
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, inputs):
        """
        Method for "__call__".

        Args:
            - inputs: List of C3, C4, C5 activation maps from backbone.

        Returns:
            List of pyramid feature maps from P3 to P7.
        """
        C3, C4, C5 = inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)


        P3_x = self.P3_1(C3)
        P3_x = P4_upsampled_x + P3_x
        P3_x = self.P3_2(P3_x)


        return P3_x, P4_x, P5_x
