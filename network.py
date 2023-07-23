import torch
import torch.nn as nn
from rnn_layer import RNNLayer

class RNNModel(nn.Module):
    """
    Recurrent neural network consisting of a continous-time RNN layer and a linear layer.
    
    N: Number of recurrently connected units
    N_in: N of input neurons
    N_out: Number of output neurons
    T: Sequence length
    M: Mini-batch size
    """

    def __init__(self, hidden_size, dt=0.02, tau=0.1):

        super(RNNModel, self).__init__()

        self.input_size = 2 # velocity, head direction
        self.hidden_size = hidden_size
        self.output_size = 2
        self.dt = dt
        self.tau = tau

        # define layers
        self.rnn = RNNLayer(self.input_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, I, x0):
        """
        Simulation of the network.

        Inputs:
            I: (M x N_in x T) Torch tensor
                Input trajectories
            x0: Scalar
                Initial value of the simulation 

        Outputs:
            x: (M x N x T+1) Torch tensor
                Activation of the units, additional +1 time due to initialization
            u: (M x N x T+1) Torch tensor
                Firing activitiy of the units, additional +1 time due to initialization
            y: (M x N_out x T) Torch tensor
                Output of the network
        """

        T = I.size(2)
        x = torch.zeros_like(I.size(0), self.hidden_size, T + 1)
        u = torch.zeros_like(x)
        y = torch.zeros(I.size(0), self.output_size, T)

        # initialization
        x[:, :, 0] = x0
        u[:, :, 0] = torch.tanh(x[:, :, 0])
        
        # simulate network
        for t in range(T):
            x[:, :, t+1], u[:, :, t+1] = self.rnn.forward_euler(x[:, :, t], I[:, :, t], self.dt, self.tau)
            y[:, t] = self.linear(u[:, :, t+1])
        
        return x, u, y
