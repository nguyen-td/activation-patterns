import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
import time
import sys

from more_itertools import chunked
from matplotlib import pyplot as plt

from network import RNNModel
from trajectory_generator import TrajectoryGenerator


class Trainer:
    """
    Train a recurrent neural network and optimize it using backpropagation. 

    Inputs:
        train_data: (batch_size x 2 x T) Numpy array
            Training data
        target_data: (batch_size x 2 x T) Numpy array
            Target data
        n_epochs: Scalar
            Number of epochs, default is 100
        mini_batch_size: Scalar
            Number of mini-batches, default is 64
        hidden_size: Scalar
            Number of hidden neurons, default is 100
        learning_rate: Scalar
            Learning rate of the optimization algorithm (e.g., RMSprop), default is 1e-4 as in [1]
        l2: Scalar
            Regularization parameter for L2 regularization, default is 1e-4 as in [2]
        l_fr: Scalar
            Regularization parameter for the metabolic cost, default is 1e-4

    [1] Sorscher, B., Mel, G. C., Ocko, S. A., Giocomo, L. M., & Ganguli, S. (2023). A unified theory for the computational and mechanistic origins of grid cells. Neuron, 111(1), 121-137. \n
    [2] Cueva, C. J., & Wei, X. X. (2018). Emergence of grid-like representations by training recurrent neural networks to perform spatial localization. arXiv preprint arXiv:1803.07770.
    """

    def __init__(self, train_data, target_data, n_epochs=100, mini_batch_size=64, hidden_size=100, learning_rate=1e-4, l2_rate=1e-4, fr_rate=1e-4, x0=0) -> None:
        self.train_data = train_data
        self.target_data = target_data

        self.mini_batch_size = mini_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.l2_rate = l2_rate
        self.fr_rate = fr_rate
        self.n_epochs = n_epochs
        self.x0 = x0

    def batch_padding(self, train, target):
        """
        Add padding to input data during mini-batch training by randomly sample data points without replacement. This is generally applied to the last batch
        where the actual number of data points is smaller than the batch size. This can happen when the number of data points is not evenly divisible by the
        selected mini-batch size.

        M: mini-batch size

        Input:
            train: (n_dat x 2 x T) Torch tensor, where n_dat <= M
                Training data
            target: (n_dat x 2 x T) Torch tensor, where n_dat <= M
                Target data
        Output:
            train_new: (M x 2 x T) Torch tensor
                Training data, padded to match the mini-batch size
            target_new: (M x 2 x T) Torch tensor
                Target data, padded to match the mini-batch size
        """
        
        # shuffling
        diff = self.mini_batch_size - train.size(0)
        shuf_inds = torch.randperm(train.size(0))
        train_shuf = train[shuf_inds]
        target_shuf = target[shuf_inds]

        # draw without replacement and pad data
        train_pad = train_shuf[:diff]
        target_pad = target_shuf[:diff]
        train_new = torch.cat([train, train_pad], dim=0)
        target_new = torch.cat([target, target_pad], dim=0)
        
        return train_new, target_new

    def train(self):

        # make mini-batches
        n_data = self.train_data.shape[0]
        train_batch = list((chunked(self.train_data, self.mini_batch_size)))
        target_batch = list((chunked(self.target_data, self.mini_batch_size)))
        n_batches = len(train_batch)

        model = RNNModel(self.hidden_size, self.mini_batch_size)
        model.double()
        optimizer = optim.RMSprop(model.parameters(), lr=self.learning_rate)
        model.to(self.device)
        print(model)

        train_loss_epochs = np.zeros(self.n_epochs)
        for epoch in range(self.n_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            start = time.time()
            train_loss = 0
            for batch in range(n_batches):
                # set up data
                train = torch.as_tensor(np.array(train_batch[batch]), device=self.device)
                target = torch.as_tensor(np.array(target_batch[batch]), device=self.device)

                # pad data if the number of data points is smaller than the selected mini-batch size
                if train.size(0) < self.mini_batch_size:
                    train, target = self.batch_padding(train, target)

                # clear out gradients
                model.zero_grad()

                # forward pass
                x, u, y = model.forward(train, self.x0)
                W_in = model.rnn.W_in
                W_out = model.linear.weight

                # compute error
                loss = model.loss(y.detach(), target, W_in, W_out, u, self.l2_rate, self.fr_rate)
                train_loss += loss.item()

                # compute gradient and update parameters
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item()}")

            train_loss /= n_batches
            train_loss_epochs[epoch] = train_loss
            end = time.time()

            print(f"Training loss: {train_loss}  {round(end - start, 3)} seconds for this epoch \n")

        return train_loss




