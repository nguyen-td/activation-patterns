import torch
from pathlib import Path

from trajectory_generator import TrajectoryGenerator
from network.network import RNNModel
from utils.make_train_data import make_train_data

# generate test data
T = 5  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
n_data_test = 100

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data_test)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
test = make_train_data(velocity, head_dir)

# load and evaluate model
hidden_size = 256
rnn_layer = 'custom'
rnn_model = torch.load(f'models/RNN-{hidden_size}-{rnn_layer}-model.pt')
aggregate_loss, y_pred = rnn_model.evaluate(test, position)

test_save_name = Path('models/test.pt')
y_pred_save_name = Path('models/ypred.pt')
torch.save(test, test_save_name)
torch.save(y_pred, y_pred_save_name)