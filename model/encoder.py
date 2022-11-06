import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

    def forward(self, x, edge_index):
        pass


class GNNEncoder(Encoder):
    def __init__(self, in_channels, hidden_channels, name='gcn'):
        super(GNNEncoder, self).__init__(in_channels, hidden_channels)

        if name == 'gcn':
            self.conv_layers = nn.ModuleList([
                GCNConv(self.in_channels, self.hidden_channels),
                GCNConv(self.hidden_channels, self.hidden_channels)
            ])
        elif name == 'gat':
            self.conv_layers = nn.ModuleList([
                GATConv(self.in_channels, self.hidden_channels),
                GATConv(self.hidden_channels, self.hidden_channels)
            ])
        elif name == 'gsage':
            self.conv_layers = nn.ModuleList([
                SAGEConv(self.in_channels, self.hidden_channels),
                SAGEConv(self.hidden_channels, self.hidden_channels)
            ])
        elif name == 'gin':
            self.conv_layers = nn.ModuleList([
                GINConv(nn.Linear(self.in_channels, self.hidden_channels)),
                GINConv(nn.Linear(self.hidden_channels, self.hidden_channels))
            ])
        else:
            assert False
        self.prelu = nn.PReLU(self.hidden_channels)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, edge_index):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = self.prelu(x)
                x = self.dropout(x)
        return x


def construct_gnn(name, num_features, encoder_dim):
    encoder = GNNEncoder(num_features, encoder_dim, name)
    return encoder
