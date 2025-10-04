# ddpg_her_pick.py
import numpy as np, random, math, collections, copy, torch, torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
import torch
from torch.utils.data import DataLoader, Dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------- 1.  Networks ----------

class Encoder_resnet18(nn.Module): # Wymaga wejścia 224x224
    def __init__(self, feat_dim=256):
        super().__init__()
        # Załaduj wstępnie wytrenowany model ResNet-18
        resnet = models.resnet18(pretrained=True)
        # Usuń ostatnią warstwę w pełni połączoną (fc)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # Dodatkowa warstwa w pełni połączona do redukcji wymiaru cech
        self.fc = nn.Linear(resnet.fc.in_features, feat_dim)

    def forward(self, x):
        # Normalizacja wejścia
        x = x.to(dtype=torch.float32, device=self.fc.weight.device)
        # Przekazanie przez backbone ResNet
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Spłaszczenie
        x = self.fc(x)
        return F.relu(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        return torch.relu(self.conv(x) + self.skip(x))

class Encoder(nn.Module):
    def __init__(self, in_ch=4, feat_dim=512, img_size=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ResNet-like bloki
        self.layer1 = ResidualBlock(64, 64)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        self.layer4 = ResidualBlock(256, 512, stride=2)

        # Adaptive pooling → stała wielkość
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, feat_dim)

    def forward(self, x):
        if x.ndim == 3:
            x = x.permute(2,0,1).unsqueeze(0)
        elif x.shape[-1] in (3,4,5):
            x = x.permute(0,3,1,2)

        x = x.to(dtype=torch.float32, device=self.fc.weight.device) / 255.0

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)             # (N,512,1,1)
        x = torch.flatten(x, 1)         # (N,512)
        return F.relu(self.fc(x))       # (N, feat_dim)



class Actor(nn.Module):
    def __init__(self, feat=512, act_dim=4):
        super().__init__()
        # sieć
        self.net = nn.Sequential(
            nn.Linear(feat, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )

        # # ───────── limity akcji ─────────
        # low = torch.tensor([-0.2, -0.2, .2], dtype=torch.float32)
        # high = torch.tensor([0.2, 0.2, .6], dtype=torch.float32)
        # self.register_buffer('low', low)
        # self.register_buffer('high', high)
        # self.register_buffer('mid', (high + low) / 2)
        # self.register_buffer('span', (high - low) / 2)

    def forward(self, feat):
        raw = self.net(feat)
        # scaled = self.mid + self.span * torch.tanh(raw)  # Przeskaluj do [low, high]
        return raw


class Critic(nn.Module):
    def __init__(self, feat=256, act_dim=4):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(feat+act_dim,128), nn.ReLU(),
            nn.Linear(128,1)
        )
    def forward(self, feat, act):
        q = self.q(torch.cat([feat,act],dim=1))
        return q

class Policy(nn.Module):
    def __init__(self, in_ch=4, img_size=64, feat_dim=256, act_dim=4):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, feat_dim=feat_dim, img_size=img_size)
        self.actor = Actor(feat=feat_dim, act_dim=act_dim)

    def forward(self, x):
        feat = self.encoder(x)
        action = self.actor(feat)
        return action
class RobotDataset(torch.utils.data.Dataset):
    def __init__(self, imgs, goals):
        self.imgs = imgs
        self.goals = goals

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx], dtype=torch.float32).permute(2,0,1) / 255.0
        goal = torch.tensor(self.goals[idx], dtype=torch.float32)
        return img, goal