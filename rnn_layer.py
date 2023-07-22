import torch
import torch.nn as nn

class RNNLayer(nn.Module):
    """Single custom RNN layer implementing the standard continuous-time RNN equation."""

    def __init__(self, input_size, hidden_size):
        super(RNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # initialize trainable parameters according to Cueva & Wei, 2018
        W_in = torch.empty(self.hidden_size, self.input_size)
        nn.init.normal(std = 1 / self.input_size)
        W_rec = torch.empty(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(W_rec)

        self.W_in = nn.Parameter(W_in)
        self.W_rec = nn.Parameter(W_rec)
        self.b = nn.Parameter(torch.zeros(self.hidden_size))
        self.xi = torch.normal(std = torch.full(self.hidden_size, 1 / self.input_size))

    def rnn_dynamics(self, x, I, tau):
        u = torch.tanh(x)
        x = 1 / tau * (-x + torch.matmul(self.W_rec, u) + torch.matmul(self.W_in, I) + self.b + self.xi)

        return x, u

    def forward_euler(self, x, I, dt, tau):
        x, u = x + dt * self.rnn_dynamics(x, I, tau)
        
        return x, u