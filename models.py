import torch.nn as nn
import torch.nn.functional as F


class MnistNet(nn.Module):
    def __init__(self, n_channels=3):
        super().__init__()
        self.n_channels = n_channels
        self.l1 = nn.Linear(n_channels*28*28, 400)
        self.l2 = nn.Linear(400, 400)
        self.l3 = nn.Linear(400, 1)

    def init_weights(self):
        for l in [self.l1, self.l2, self.l3]:
            nn.init.normal_(l.weight)

    def forward(self, x):
        x = x.view(x.shape[0], self.n_channels*28*28)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        return self.l3(x)

