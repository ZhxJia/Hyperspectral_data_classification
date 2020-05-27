import torch
import torch.nn as nn
from utils.hsidataset import *


class HsiNet(nn.Module):
    def __init__(self, num_class=17):
        super(HsiNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(30, 64, 3, 1, 0, bias=False),
                                   nn.BatchNorm2d(64, momentum=0.03, eps=1E-4),
                                   nn.LeakyReLU(0.01))
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 0)
        self.relu2 = nn.LeakyReLU(0.1)
        self.conv3 = nn.Conv2d(128, 128, 1, 1, 0)
        self.relu3 = nn.LeakyReLU(0.01)  # 1*1*128
        self.logits = nn.Linear(128, num_class)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = x.view(x.size(0), -1)
        x = self.logits(x)
        return x


if __name__ == "__main__":
    hsi_traindatas = HsiDataset("./data", type='train', oversampling=False)
    train_dataloader = DataLoader(hsi_traindatas, batch_size=4, shuffle=True, num_workers=2)
    hsinet = HsiNet(num_class=17)
    # x = torch.Tensor(4, 30, 5, 5)
    # y = hsinet(x)

    for i, (patch, label) in enumerate(train_dataloader):
        y = hsinet(patch)
        print(y.shape)