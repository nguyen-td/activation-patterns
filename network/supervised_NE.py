import torch
from evotorch.neuroevolution import SupervisedNE

from network.network import RNNModel

class CustomSupervisedNE(SupervisedNE):
    def __init__(self, dataset, device, minibatch_size, num_actors, hidden_size, rnn_layer, l2_rate, fr_rate, dt, tau, x0):
        
        self.rnn_layer = rnn_layer
        self.l2_rate = l2_rate
        self.fr_rate = fr_rate

        model = RNNModel(hidden_size, minibatch_size, self.rnn_layer, self.l2_rate, self.fr_rate, dt, tau, x0)
        model.double()
        model.to(device)

        super(CustomSupervisedNE, self).__init__(
            dataset = dataset, 
            device = device,
            network = model,
            minibatch_size = minibatch_size, 
            num_actors = num_actors)

    def _evaluate_using_minibatch(self, network, batch):
        """
        Pass a minibatch through a network, and compute the loss. 

        WARNING: So far, this only works for the custom RNN. 

        Inputs:
            network: nn.Module
                Network to be trained
            batch: TensorDataset
                Minibatch that will be used as data

        Output:
            E: Torch scalar
                Error value/loss
        """
        with torch.no_grad():
            input, y = batch

            if self.rnn_layer == 'custom':
                x, u, y_hat = network.forward_custom_rnn(input.to(self.device), y.to(self.device)) # forward pass
                W_in = network.rnn.W_in
            else:
                u, y_hat = network.forward_native_rnn(input)
                W_in = network.rnn.weight_ih_l0
            W_out = network.linear.weight

            return self._loss(y_hat, y.to(self.device), W_in, W_out, u)

    def _loss(self, y_hat, y, W_in, W_out, u):
        """
        Compute loss by minimizing the MSE between target and prediction while also minimizing the metabolic 
        cost and penalizing large network parameters (L2 weight decay). It is technically a duplicate definition 
        (it is already defined in network.network) but the SupervisedNE method requires a separate definition.

        Inputs:
            y_hat: (M x T x N_out) Torch tensor
                Output of the network
            y: (M x T x N_out) Torch tensor
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
        E = torch.mean((y_hat - y)**2) + self.l2_rate * R_l2 + self.fr_rate * R_fr # minimize error of animal
        return E