import torch
import torch.nn as nn


class RNNLayer(nn.Module):
    """
    Single custom RNN layer implementing the standard continuous-time RNN equation.

    N: Number of recurrently connected units
    N_in: N of input neurons
    N_out: Number of output neurons
    T: Sequence length
    M: Mini-batch size
    """

    def __init__(self, input_size, hidden_size, batch_size, activation, device):
        super(RNNLayer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.device = device
        self.activation = activation

        # initialize trainable parameters according to Cueva & Wei, 2018
        W_in = torch.empty(self.hidden_size, self.input_size)
        nn.init.normal_(W_in, std = 1 / self.input_size)
        W_rec = torch.empty(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(W_rec)

        self.W_in = nn.Parameter(W_in)
        self.W_rec = nn.Parameter(W_rec)
        self.b = nn.Parameter(torch.zeros(self.hidden_size, 1))
        self.xi = torch.normal(mean = 0, std = torch.full((self.hidden_size, 1), 1 / self.input_size)).to(device=self.device) # not trained

    def rnn_dynamics(self, x, I, tau):
        """
        Dynamics of the units in the layer.

        Inputs: 
            x: (M x N) Torch tensor
                Activation of the units
            I: (M x N_in) Torch tensor
                Input to the layer
            tau: Scalar
                Time scale of decay

        Outputs:
            x: (M x N) Torch tensor
                Activation of the units
            u: (M x N) Torch tensor
                Firing activity of the units
        """
        
        # expand terms for matrix operation with batches
        b = self.b 
        b = torch.t(b.expand(-1, self.batch_size)).double()
        xi = self.xi
        xi = torch.t(xi.expand(-1, self.batch_size)).double()
        
        if self.activation == 'relu':
            u = torch.relu(x)
        elif self.activation == 'heaviside': # heaviside step function
            u = torch.heaviside(x, torch.tensor([1]))
        else:
            u = torch.tanh(x)
        x = 1 / tau * (-x + torch.matmul(u, self.W_rec) + torch.matmul(I, torch.t(self.W_in)) + b + xi)

        return x, u

    def forward_euler(self, x_in, I, dt, tau):
        """
        Forward Euler integration of the differential equation.

        Inputs: 
            x_in: (M x N) Torch tensor
                Activation of a unit
            I: (M x N_in) Torch tensor
                Input to the layer
            dt: Scalar
                Step size
            tau: Scalar
                Time scale of decay
        Outputs:
            x_out: (M x N) Torch tensor
                Activation of the units
            u: (M x N) Torch tensor
                Firing activity of the units

        """
        
        x_out, u = self.rnn_dynamics(x_in, I, tau)
        x = x_in + dt * x_out

        return x, u