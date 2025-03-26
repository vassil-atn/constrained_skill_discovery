import copy
import numpy as np
import torch
from torch import nn

# def init_weights_he(m):
#     if isinstance(m, nn.Linear):
#         nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)

class TrajectoryEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,min_log_std=-5,max_log_std=2, use_spectral_norm=False, use_dropout=False,dropout_rate=0.1,use_batch_norm=False,device='cpu'):
        super(TrajectoryEncoder, self).__init__()

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.use_spectral_norm = use_spectral_norm

        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        layers = []

        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(input_dim, hidden_dim[0])))
        else:
            layers.append(nn.Linear(input_dim, hidden_dim[0]))
        layers.append(nn.ReLU())

        for i in range(1, len(hidden_dim)):
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(hidden_dim[i-1], hidden_dim[i])))
            else:
                layers.append(nn.Linear(hidden_dim[i-1], hidden_dim[i]))
            layers.append(nn.ReLU())
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim[i]))
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))

        # output layers (mean and std):
        if use_spectral_norm:
            self.mean_output = nn.utils.spectral_norm(nn.Linear(hidden_dim[-1], output_dim)).to(device)
            self.log_std_output = nn.utils.spectral_norm(nn.Linear(hidden_dim[-1], output_dim)).to(device)
        else:
            self.mean_output = nn.Linear(hidden_dim[-1], output_dim).to(device)
            self.log_std_output = nn.Linear(hidden_dim[-1], output_dim).to(device)

        self.device = device
        self.encoder = nn.Sequential(*layers).to(device)

    def forward(self, x):
        return self.mean_output(self.encoder(x)), self.log_std_output(self.encoder(x))
    
    def get_distribution(self, x):
        mean, log_std = self.forward(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        distr = torch.distributions.Normal(mean, log_std.exp())

        return distr
    