import torch
print(torch.cuda.is_available())
import matplotlib.pyplot as plt
from pathlib import Path

from trainers.trainer import Trainer
from utils import make_train_data, TrajectoryGenerator
from network.network import RNNModel

# generate training data
T = 10 * 60  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 16
n_data = mini_batch_size * 1000
activation = 'relu'

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
torch.save([position[0], velocity[0], head_dir[0]], f'data-{activation}.pt')
train = make_train_data(velocity, head_dir)

# start training
n_epochs = 20
hidden_size = 256
rnn_layer = 'custom'
model_name = f'RNN-{hidden_size}-{rnn_layer}-{activation}'

trainer = Trainer(train, position, model_name, hidden_size=hidden_size, mini_batch_size=mini_batch_size, rnn_layer=rnn_layer, n_epochs=n_epochs, activation=activation)
train_loss_epochs = trainer.train()

# plot training progress
plt.plot(train_loss_epochs)
plt.title('Training loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png', bbox_inches='tight')

# plt.show()