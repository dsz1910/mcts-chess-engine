import torch
from torch import nn


class ChessNet(nn.Module):

    def __init__(self, num_res_blocks=10) -> None:
        super().__init__()
        torch.backends.cudnn.benchmark = True
        
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.backbone = nn.ModuleList([ResidualBlock(128) for i in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Flatten(),
            #nn.Linear(32_768, 4096),
            nn.Linear(32_768, 1024),
            #nn.Dropout(0.3),
            #nn.LeakyReLU(),
            #nn.Linear(4096, 1024),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
            nn.Linear(1024, 1972)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Flatten(),
            #nn.Linear(2048, 512),
            #nn.Dropout(0.5),
            #nn.LeakyReLU(),
            #nn.Linear(512, 1),
            nn.Linear(2048, 1),
            nn.Tanh()
        )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.lrelu(x)
        x = self.bn2(self.conv2(x))
        x = self.lrelu(x)

        for block in self.backbone:
            x = block(x)

        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value

class ResidualBlock(nn.Module):

    def __init__(self, input) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input, input, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(input)
        self.lrelu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(input, input, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(input)

    def forward(self, x):
        residual = x
        x = self.bn1(self.conv1(x))
        x = self.lrelu(x)
        x = self.bn2(self.conv2(x))
        x += residual
        x = self.lrelu(x)
        return x