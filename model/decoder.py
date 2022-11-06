import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        pass


class NNDecoder(Decoder):
    def __init__(self, hidden_dim, output_dim, layer_num):
        super().__init__(hidden_dim, output_dim)
        self.nns = [nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        ) for _ in range(layer_num - 1)]
        self.nns.append(nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        ))

    def forward(self, x):
        for i in self.nns:
            x = i(x)
        return x

