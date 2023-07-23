import torch
import torch.nn as nn
from network import RNNModel
from trajectory_generator import TrajectoryGenerator
from more_itertools import chunked
import torch.optim as optim
from torchsummary import summary

def loss_fn(pred, y, W_in, W_out, u, l_l2,l_fr):
    R_l2 = torch.mean(W_in**2) + torch.mean(W_out**2) # L2 regularization to penalize large network parameters
    R_fr = torch.mean(u**2) # minimize metabolic cost
    E = torch.mean((pred - y)**2) + l_l2 * R_l2 + l_fr * R_fr # minimize error of animal
    return E

# generate training data
T = 20  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 64
n_data = mini_batch_size * 30

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()

# make mini-batches
pos_batch = list(chunked(position, mini_batch_size))
vel_batch = list(chunked(velocity, mini_batch_size))
hd_batch = list(chunked(head_dir, mini_batch_size))

# TODO: put velocity and hd batches together to make a (M x 2 x T) input tensor

# initialize model
hidden_size = 100 # Cueva et al., 2018
learning_rate = 1e-4 # Sorscher et al., 2023
l2_rate = 1e-4 # Sorscher et al., 2023
fr_rate = 1e-4
device = 'cpu'
n_epochs = 10
# device = 'cuda:0'

model = RNNModel(hidden_size)
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
model.to(device)
print(model)
# summary(model, (mini_batch_size, 2, sequence_length))

for t in range(n_epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    for batch in range(mini_batch_size):
        # clear out gradients
        model.zero_grad()

        # forward pass
        x, u, y = model.forward()


