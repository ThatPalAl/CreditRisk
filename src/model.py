import torch.nn as nn

class LinearCredit(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)

    def forward(self, x):
        return self.lin(x)


class MlpCredit(nn.Module):
    def __init__(self, d_in, hidden=64, p=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)