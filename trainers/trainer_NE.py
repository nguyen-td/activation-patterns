import torch 
from torch.utils.data import TensorDataset
from evotorch.algorithms import SNES
from evotorch.logging import PandasLogger, StdOutLogger

import matplotlib.pyplot as plt

from network.supervised_NE import CustomSupervisedNE

class Trainer_NE:
    """
    Train a recurrent neural network and optimize it using neuroevolution. 

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
        num_actors: Scalar
            Number of Actors (EvoTorch)
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
    """

    def __init__(self, train_data, target_data, model_name, rnn_layer='native', n_epochs=100, mini_batch_size=64, hidden_size=100, num_actors=4, l2_rate=1e-4, fr_rate=1e-4, dt=0.02, tau=0.1, x0=0) -> None:
        self.train_dataset = TensorDataset(torch.as_tensor(train_data), torch.as_tensor(target_data))
        self.model_name = model_name
        self.rnn_layer = rnn_layer

        self.mini_batch_size = mini_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

        # model parameters
        self.hidden_size = hidden_size
        self.l2_rate = l2_rate
        self.fr_rate = fr_rate
        self.n_epochs = n_epochs
        self.num_actors = num_actors

        # simulation parameters
        self.x0 = x0
        self.dt = dt 
        self.tau = tau
        
    def train(self):

        # create problem class, the model is defined within this class
        problem = CustomSupervisedNE(
            dataset = self.train_dataset,
            device = self.device,
            minibatch_size = self.mini_batch_size,
            num_actors = self.num_actors,
            hidden_size = self.hidden_size,
            rnn_layer = self.rnn_layer,
            l2_rate = self.l2_rate,
            fr_rate = self.fr_rate,
            dt = self.dt,
            tau = self.tau,
            x0 = self.x0
        )

        searcher = SNES(problem, popsize = 10, radius_init = 2.25)
        stdout_logger = StdOutLogger(searcher, interval = 1)
        pandas_logger = PandasLogger(searcher, interval = 1)
        searcher.run(self.n_epochs)
        pandas_logger.to_dataframe().mean_eval.plot()
        plt.show() # save figure instead

        # safe searcher?