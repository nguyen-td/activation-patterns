import torch
print(torch.cuda.is_available())
import matplotlib.pyplot as plt
from pathlib import Path

from trainers.trainer import Trainer
from trajectory_generator import TrajectoryGenerator
from utils import make_train_data
from network.network import RNNModel

# generate training data
T = 5  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 32
n_data = mini_batch_size * 1000

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
torch.save(position, Path('models/y_true_train.pt'))
train = make_train_data(velocity, head_dir)

# start training
n_epochs = 100
hidden_size = 256
rnn_layer = 'custom'
model_name = f'RNN-{hidden_size}-{rnn_layer}'

trainer = Trainer(train, position, model_name, hidden_size=hidden_size, mini_batch_size=mini_batch_size, rnn_layer=rnn_layer, n_epochs=n_epochs)
train_loss_epochs = trainer.train()

# plot training progress
plt.plot(train_loss_epochs)
plt.title('Training loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png', bbox_inches='tight')

# plt.show()