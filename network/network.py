import torch
import torch.nn as nn
import time
import numpy as np
from more_itertools import chunked
from pathlib import Path

from .rnn_layer import RNNLayer
from utils import batch_padding


class RNNModel(nn.Module):
    """
    Recurrent neural network consisting of a continous-time RNN layer and a linear layer.
    
    N: Number of recurrently connected units
    N_in: N of input neurons
    N_out: Number of output neurons
    T: Sequence length
    M: Mini-batch size

    Inputs:
        rnn_layer: String {'custom', 'native'}, default: 'native'
            Type of RNN layer to use. If 'custom' is chosen, the custom layer will be used. Else, PyTorch's native RNN layer will be used.
        l2_rate: Scalar
            Regularization parameter for L2 regularization
        fr_rate: Scalar
            Regularization parameter for the metabolic cost
        tau: Scalar
            Time scale of decay
        x0: Scalar
            Initial value of the simulation 
    """

    def __init__(self, hidden_size, batch_size, rnn_layer='native', l2_rate=1e-4, fr_rate=1e-4, dt=0.02, tau=0.1, x0=0):
        super(RNNModel, self).__init__()

        self.input_size = 2 # velocity, head direction
        self.hidden_size = hidden_size
        self.output_size = 2
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        self.rnn_layer = rnn_layer

        # simulation parameters
        self.dt = dt
        self.tau = tau
        self.x0 = x0

        # define layers
        if rnn_layer == 'custom':
            self.rnn = RNNLayer(self.input_size, self.hidden_size, self.batch_size)
        else:
            self.rnn = nn.RNN(self.input_size, self.hidden_size, bias=True, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

        # loss function parameters
        self.l2_rate = l2_rate
        self.fr_rate = fr_rate

    def loss(self, y_pred, y_true, W_in, W_out, u):
        """
        Compute loss by minimizing the MSE between target and prediction while also minimizing the metabolic 
        cost and penalizing large network parameters (L2 weight decay).

        Inputs:
            y_pred: (M x T x N_out) Torch tensor
                Output of the network
            y_true: (M x T x N_out) Torch tensor
                Ground truth targets
            W_in: (N x N_in) Torch tensor
                Hidden-to-input weight matrix
            W_out: (N_out x N) Torch tensor
                Hidden-to-output weight matrix
            u: (M x T x N) Torch tensor
                Firing activity of the units
        
        Output:
            E: Torch scalar 
                Error value/loss
        """

        R_l2 = torch.mean(W_in**2) + torch.mean(W_out**2) # L2 regularization 
        R_fr = torch.mean(u**2) # minimize metabolic cost
        E = torch.mean((y_pred - y_true)**2) + self.l2_rate * R_l2 + self.fr_rate * R_fr # minimize error of animal
        return E

    def forward_native_rnn(self, I):
        u, _ = self.rnn(I)
        y = self.linear(u)

        return u, y

    def forward_custom_rnn(self, I):
        """
        Wrapper function for the simulation of the network.

        Inputs:
            I: (M x T x N_in) Torch tensor
                Input trajectories

        Outputs:
            x: (M x T x N) Torch tensor
                Activation of the units
            u: (M x T x N) Torch tensor
                Firing activitiy of the units
            y: (M x T x N_out) Torch tensor
                Output of the network
        """

        T = I.size(1)
        x = torch.zeros(self.batch_size, T, self.hidden_size, dtype=torch.double, device=self.device)
        u = torch.zeros_like(x, device=self.device)
        y = torch.zeros(self.batch_size, T, self.output_size, device=self.device)

        # initialization
        x[:, 0, :] = torch.tensor(self.x0, device=self.device)
        u[:, 0, :] = torch.tanh(x[:, 0, :])
        
        # simulate network
        for t in range(T-1):
            x[:, t+1, :], u[:, t+1, :] = self.rnn.forward_euler(x[:, t, :], I[:, t, :], self.dt, self.tau)
        y = self.linear(u)
        
        return x, u, y
    
    def evaluate(self, input_test, target_test):
        """
        Run the trained model on test data to path integrate from velocity and head direction inputs. 

        Inputs:
            input_test: (n_dat x T x N_in) Torch tensor
                Input trajectories
            target_test: (n_dat x T x N_out) Torch tensor
                Target output trajectories
        Outputs:
            aggregate_loss: Scalar
                Aggregated loss
            y_pred: (n_dat x T x N_out) Torch tensoor
                Predicted output trajectories
        """

        # make mini-batches
        input_batch = list((chunked(input_test, self.batch_size)))
        target_batch = list((chunked(target_test, self.batch_size)))
        
        n_batches = len(input_batch)
        n_data = input_test.shape[0]

        start = time.time()
        aggregate_loss = 0

        y_pred = list()
        x_test = list()
        with torch.no_grad():
            print('Start evaluation run: ')
            for batch in range(n_batches):
                print('.', end='')

                # set up data
                input = torch.as_tensor(np.array(input_batch[batch]), device=self.device)
                target = torch.as_tensor(np.array(target_batch[batch]), device=self.device)

                # pad data if the number of data points is smaller than the selected mini-batch size
                if input.size(0) < self.batch_size:
                    input, target = batch_padding(input, target, self.batch_size)

              # forward pass
                if self.rnn_layer == 'custom':
                    x, u, y = self.forward_custom_rnn(input)
                    W_in = self.rnn.W_in
                else:
                    u, y = self.forward_native_rnn(input)
                    W_in = self.rnn.weight_ih_l0
                W_out = self.linear.weight

                # compute error
                loss = self.loss(y, target, W_in, W_out, u)
                aggregate_loss += loss.item()

                # save prediction
                y_pred.append(y.detach().cpu().numpy())
                x_test.append(x.detach().cpu().numpy())

            aggregate_loss /= n_batches
            end = time.time()
            print("\n")
            print(f"Aggregated loss: {aggregate_loss}  {round(end - start, 3)} seconds for this run \n")

            # get prediction and model activitiy
            y_pred = [item for sublist in y_pred for item in sublist]
            y_pred = np.array(y_pred[:n_data])
            x_test = [item for sublist in x_test for item in sublist]
            x_test = np.array(x_test[:n_data])

            # save stuff
            y_pred_save_name = Path('models/y_pred_train.pt')
            x_test_save_name = Path('models/x_test.pt')
            # torch.save(y_pred, y_pred_save_name)
            # torch.save(x_test, x_test_save_name)

        return aggregate_loss, x_test, y_pred


            