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

    def __init__(self, hidden_size, batch_size, dt=0.02, tau=0.1):
        super(RNNModel, self).__init__()

        self.input_size = 2 # velocity, head direction
        self.hidden_size = hidden_size
        self.output_size = 2
        self.batch_size = batch_size
        self.dt = dt
        self.tau = tau

        # define layers
        self.rnn = RNNLayer(self.input_size, self.hidden_size, self.batch_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def loss(self, y_pred, y_true, W_in, W_out, u, l_l2, l_fr):
        """
        Compute loss by minimizing the MSE between target and prediction while also minimizing the metabolic 
        cost and penalizing large network parameters (L2 weight decay).

        Inputs:
            y_pred: (M x N_out x T) Torch tensor
                Output of the network
            y_true: (M x N_out x T) Torch tensor
                Ground truth targets
            W_in: (N x N_in) Torch tensor
                Hidden-to-input weight matrix
            W_out: (N_out x N) Torch tensor
                Hidden-to-output weight matrix
            u: (M x N x T) Torch tensor
                Firing activity of the units
            l2: Scalar
                Regularization parameter for L2 regularization
            l_fr: Scalar
                Regularization parameter for the metabolic cost
        
        Output:
            E: Torch scalar 
                Error value
        """

        R_l2 = torch.mean(W_in**2) + torch.mean(W_out**2) # L2 regularization 
        R_fr = torch.mean(u**2) # minimize metabolic cost
        E = torch.mean((y_pred - y_true)**2) + l_l2 * R_l2 + l_fr * R_fr # minimize error of animal
        return E

    def forward(self, I, x0):
        """
        Simulation of the network.

        Inputs:
            I: (M x N_in x T) Torch tensor
                Input trajectories
            x0: Scalar
                Initial value of the simulation 

        Outputs:
            x: (M x N x T) Torch tensor
                Activation of the units
            u: (M x N x T) Torch tensor
                Firing activitiy of the units
            y: (M x N_out x T) Torch tensor
                Output of the network
        """

        T = I.size(2)
        x = torch.zeros(self.batch_size, self.hidden_size, T + 1, dtype=torch.double)
        u = torch.zeros_like(x)
        y = torch.zeros(self.batch_size, self.output_size, T)

        # initialization
        x[:, :, 0] = x0
        u[:, :, 0] = torch.tanh(x[:, :, 0])
        
        # simulate network
        for t in range(T):
            x[:, :, t+1], u[:, :, t+1] = self.rnn.forward_euler(x[:, :, t], I[:, :, t], self.dt, self.tau)
            y[:, :, t] = self.linear(u[:, :, t+1])
        
        # drop last time step
        x = x[:, :, :-1]
        u = u[:, :, :-1]
        return x, u, y
