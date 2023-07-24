import torch
import torch.nn as nn
import torch.optim as optim
from torch import autograd
import numpy as np
import time

from more_itertools import chunked
from matplotlib import pyplot as plt

from network import RNNModel
from trajectory_generator import TrajectoryGenerator


# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0

# generate training data
T = 20  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 64
n_data = mini_batch_size * 2

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()

# make training and target (ground truth) data
training_data = np.stack((velocity, head_dir), axis=1)
target_data = np.transpose(position, (0, 2, 1))

# make mini-batches
train_batch = torch.as_tensor(np.array(list((chunked(training_data, mini_batch_size)))), device=device)
target_batch = torch.as_tensor(np.array(list((chunked(target_data, mini_batch_size)))), device=device)

# initialize model
hidden_size = 100 # Cueva et al., 2018
learning_rate = 1e-4 # Sorscher et al., 2023
l2_rate = 1e-4 # Sorscher et al., 2023
fr_rate = 1e-4
n_epochs = 3
n_batches = int(n_data / mini_batch_size)
x0 = 0

model = RNNModel(hidden_size, mini_batch_size)
model.double()
optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
model.to(device)
print(model)

train_loss_epochs = np.zeros(n_epochs)
for epoch in range(n_epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    start = time.time()
    train_loss = 0
    for batch in range(n_batches):

        # clear out gradients
        model.zero_grad()

        # forward pass
        x, u, y = model.forward(train_batch[batch], x0)
        W_in = model.rnn.W_in
        W_out = model.linear.weight

        # compute error
        loss = model.loss(y.detach(), target_batch[batch], W_in, W_out, u, l2_rate, fr_rate)
        train_loss += loss.item()

        # compute gradient and update parameters
        loss.backward()
        optimizer.step()
        print(f"loss: {train_loss}")

    train_loss /= n_batches
    train_loss_epochs[epoch] = train_loss
    end = time.time()

    print(f"Training loss: {train_loss}  {round(end - start, 3)} seconds for this epoch \n")

plt.plot(train_loss)
plt.show()




