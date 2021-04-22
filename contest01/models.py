import torchvision.models as models
import torch.nn as nn
import torch


class MyResNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(MyResNet, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.requires_grad_(False)

        fc_input = 512 * (input_size // 2 ** 5) ** 2
        self.model.fc = nn.Linear(fc_input, output_size, bias=True)
        self.model.fc.requires_grad_(True)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x
