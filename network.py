import torch
import torch.nn as nn
from rnn_layer import RNNLayer

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.02, tau=0.1):
        super(Model, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        self.dt = dt
        self.tau = tau

        # define layers
        self.rnn = RNNLayer(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, x, I):
        x, u = self.rnn.forward_euler(x, I, self.dt, self.tau),float()
        y = self.linear(u)
        
        return x, u, y
