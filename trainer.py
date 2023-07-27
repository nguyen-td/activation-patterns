import torch
import torch.optim as optim
import numpy as np
import time
from pathlib import Path
from more_itertools import chunked

from network import RNNModel
from utils.batch_padding import batch_padding


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

    def __init__(self, train_data, target_data, model_name, n_epochs=100, mini_batch_size=64, hidden_size=100, learning_rate=1e-4, l2_rate=1e-4, fr_rate=1e-4, dt=0.02, tau=0.1, x0=0) -> None:
        self.train_data = train_data
        self.target_data = target_data
        self.model_name = model_name

        self.batch_size = mini_batch_size
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
        train_batch = list((chunked(self.train_data, self.batch_size)))
        target_batch = list((chunked(self.target_data, self.batch_size)))
        n_batches = len(train_batch)

        model = RNNModel(self.hidden_size, self.batch_size, self.l2_rate, self.fr_rate, self.dt, self.tau, self.x0)
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
                if train.size(0) < self.batch_size:
                    train, target = batch_padding(train, target, self.batch_size)

                # clear out gradients
                model.zero_grad()

                # forward pass
                x, u, y = model.forward(train)
                W_in = model.rnn.W_in
                W_out = model.linear.weight

                # compute error
                loss = model.loss(y.detach(), target, W_in, W_out, u)
                train_loss += loss.item()

                # compute gradient and update parameters
                loss.backward()
                optimizer.step()
                print(f"loss: {loss.item()}")

            train_loss /= n_batches
            train_loss_epochs[epoch] = train_loss
            end = time.time()

            model_save_name = Path('models') / f'{self.model_name}-model.pt'
            torch.save(model, model_save_name)

            print(f"Training loss: {train_loss}  {round(end - start, 3)} seconds for this epoch \n")

        return train_loss_epochs




