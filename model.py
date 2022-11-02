import torch
from torch import nn
from torch.nn import utils

from render import render

# DeepSDF network
class DeepSDF(nn.Module):
    def __init__(self, num_layers=8, hidden_dim=512, use_weight_norm=True,
                 use_dropout=True, dropout_prob=0.2):
        super().__init__()
        self.first_fc = nn.Linear(3, hidden_dim)

        self.inter_fc = nn.ModuleList()
        num_inter_layers = num_layers - 2
        self.skip_layer = num_inter_layers // 2

        for i in range(num_inter_layers):
            if i == self.skip_layer:  # Skip connection
                out_dim = hidden_dim - 3
            else:
                out_dim = hidden_dim
            self.inter_fc.append(nn.Linear(hidden_dim, out_dim))

        self.last_fc = nn.Linear(hidden_dim, 1)

        if use_weight_norm:
            self.apply_weight_norm()

        if use_dropout:
            self.activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout_prob))
        else:
            self.activation = nn.ReLU()

    def apply_weight_norm(self):
        self.first_fc = utils.weight_norm(self.first_fc)
        self.last_fc = utils.weight_norm(self.last_fc)
        for i, layer in enumerate(self.inter_fc):
            self.inter_fc[i] = utils.weight_norm(layer)

    def forward(self, coords, use_clip: bool):
        # coords: 3D point coordinates. [B, 3]
        out = self.activation(self.first_fc(coords))  # [B, H]

        for i, inter_layer in enumerate(self.inter_fc):
            out = self.activation(inter_layer(out))
            if i == self.skip_layer:
                out = torch.cat([out, coords], dim=-1)  # [B, H - 3] -> [B, H]

        out = self.last_fc(out)  # [B, 1]
        out = torch.tanh(out)

        if use_clip:
            out = render(out)
        return out
