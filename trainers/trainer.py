import torch
from torch import nn
import torch.optim as optim
import numpy as np
import time
import os
from pathlib import Path
from more_itertools import chunked

from network.network import RNNModel
from utils import batch_padding


class Trainer:
    """
    Train a recurrent neural network and optimize it using backpropagation. 

    Inputs:
        train_data: (batch_size x T x 2) Numpy array
            Training data
        target_data: (batch_size x T x 2) Numpy array
            Target data
        model_name: String
            Name of the saved model
        rnn_layer: String {'custom', else}
            Type of RNN layer to use. If 'custom' is chosen, the custom layer will be used. Else, PyTorch's native RNN layer will be used.
        n_epochs: Scalar
            Number of epochs, default is 100
        mini_batch_size: Scalar
            Number of mini-batches, default is 64
        hidden_size: Scalar
            Number of hidden neurons, default is 100
        learning_rate: Scalar
            Learning rate of the optimization algorithm (e.g., RMSprop), default is 1e-4 as in [1]
        l2_rate: Scalar
            Regularization parameter for L2 regularization, default is 1e-4 as in [2]
        fr_rate: Scalar
            Regularization parameter for the metabolic cost, default is 1e-4
        dt: Scalar
            Step size
        tau: Scalar
            Time scale of decay
        x0: Scalar
            Initial value of the simulation

    [1] Sorscher, B., Mel, G. C., Ocko, S. A., Giocomo, L. M., & Ganguli, S. (2023). A unified theory for the computational and mechanistic origins of grid cells. Neuron, 111(1), 121-137. \n
    [2] Cueva, C. J., & Wei, X. X. (2018). Emergence of grid-like representations by training recurrent neural networks to perform spatial localization. arXiv preprint arXiv:1803.07770.
    """

    def __init__(self, train_data, target_data, model_name, rnn_layer='native', n_epochs=100, mini_batch_size=64, hidden_size=100, learning_rate=1e-4, l2_rate=1e-4, fr_rate=1e-4, dt=0.02, tau=0.1, x0=0) -> None:
        self.train_data = train_data
        self.target_data = target_data
        self.model_name = model_name
        self.rnn_layer = rnn_layer

        self.mini_batch_size = mini_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

        # model parameters
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate
        self.fr_rate = fr_rate
        self.n_epochs = n_epochs

        # simulation parameters
        self.x0 = x0
        self.dt = dt 
        self.tau = tau

    def train(self):

        # make mini-batches
        train_batch = list((chunked(self.train_data, self.mini_batch_size)))
        target_batch = list((chunked(self.target_data, self.mini_batch_size)))
        n_batches = len(train_batch)
        n_data = self.train_data.shape[0]

        model = RNNModel(self.hidden_size, self.mini_batch_size, self.rnn_layer, self.l2_rate, self.fr_rate, self.dt, self.tau, self.x0)
        model.double()
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        model.to(self.device)
        print(model)

        # save parameters
        save_folder = 'models'
        if not os.path.isdir(save_folder):
            os.mkdir(save_folder)

        train_loss_epochs = np.zeros(self.n_epochs)
        y_pred = list()
        x_train = list()
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            start = time.time()
            train_loss = 0
            for batch in range(n_batches):
                # set up data
                train = torch.as_tensor(np.array(train_batch[batch]), device=self.device)
                target = torch.as_tensor(np.array(target_batch[batch]), device=self.device)

                # pad data if the number of dat a points is smaller than the selected mini-batch size
                # FIXME: fix this by resampling from the whole data set
                if train.size(0) < self.mini_batch_size:
                    train, target = batch_padding(train, target, self.mini_batch_size)

                # clear out gradients
                model.zero_grad()

                # forward pass
                if self.rnn_layer == 'custom':
                    x, u, y = model.forward_custom_rnn(train)
                    W_in = model.rnn.W_in
                else:
                    u, y = model.forward_native_rnn(train)
                    W_in = model.rnn.weight_ih_l0
                W_out = model.linear.weight

                # compute error
                loss = model.loss(y, target, W_in, W_out, u)
                train_loss += loss.item()

                # compute gradient and update parameters
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item()}")

                # save prediction and model activity in the last iteration
                if epoch == (self.n_epochs - 1):
                    y_pred.append(y.detach().cpu().numpy())
                    x_train.append(x.detach().cpu().numpy())

            train_loss /= n_batches
            train_loss_epochs[epoch] = train_loss
            end = time.time()

            print(f"Training loss: {train_loss}  {round(end - start, 3)} seconds for this epoch \n")

            # save in-between
            model_save_name = Path('models') / f'{self.model_name}-{epoch}-model.pt'
            torch.save(model.state_dict(), model_save_name)

        # save stuff
        # y_pred = [item for sublist in y_pred for item in sublist]
        # y_pred = np.array(y_pred[:n_data])
        # x_train = [item for sublist in x_train for item in sublist]
        # x_train = np.array(x_train[:n_data])

        model_save_name = Path('models') / f'{self.model_name}-model.pt'
        # y_pred_save_name = Path('models/y_pred_train.pt')
        # x_train_save_name = Path('models/x_train.pt')
        # torch.save(model.state_dict(), model_save_name)
        torch.save(model, model_save_name)
        # torch.save(y_pred, y_pred_save_name)
        # torch.save(x_train, x_train_save_name)

        return train_loss_epochs




