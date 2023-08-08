import torch
print(torch.cuda.is_available())
import matplotlib.pyplot as plt
from pathlib import Path

from trainers.trainer import Trainer
from trajectory_generator import TrajectoryGenerator
from utils import make_train_data
from network.network import RNNModel

# generate training data
T = 20  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 32
n_data = mini_batch_size * 314

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
torch.save(position, Path('models/y_true_train.pt'))
train = make_train_data(velocity, head_dir)

# start training
n_epochs = 3
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



# generate test data
n_data_test = mini_batch_size * 3
trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data_test)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
test = make_train_data(velocity, head_dir)

# rnn_model = RNNModel(hidden_size, mini_batch_size, rnn_layer)
# rnn_model.double()
rnn_model = torch.load(f'models/RNN-{hidden_size}-{rnn_layer}-model.pt')
rnn_model.eval()
aggregate_loss, y_pred, x = rnn_model.evaluate(test, position)

# plot trajectories generated from the network (on test data)
traj_idx = 3
plt.scatter(position[traj_idx, 0, 0], position[traj_idx, 0, 1], color = 'red', label = 'simulated starting point')
plt.plot(position[traj_idx, :, 0], position[traj_idx, :, 1], label = 'simulated trajectory')

plt.scatter(y_pred[traj_idx, 0, 0], y_pred[traj_idx, 0, 1], color = 'green', label = 'decoded starting point')
plt.plot(y_pred[traj_idx, :, 0], y_pred[traj_idx, :, 1], label = 'decoded trajectory')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.legend(bbox_to_anchor=(1.5, 1.))

plt.savefig('trajectory1.png', bbox_inches='tight')
# plt.show()

# second model, load from state dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
rnn_model = RNNModel(hidden_size, mini_batch_size, rnn_layer)
rnn_model.double()
rnn_model.to(device)
rnn_model.load_state_dict(torch.load(f'models/RNN-{hidden_size}-{rnn_layer}-model_statedict.pt'))
rnn_model.eval()
aggregate_loss, y_pred, x = rnn_model.evaluate(test, position)

# plot trajectories generated from the network (on test data)
traj_idx = 3
plt.scatter(position[traj_idx, 0, 0], position[traj_idx, 0, 1], color = 'red', label = 'simulated starting point')
plt.plot(position[traj_idx, :, 0], position[traj_idx, :, 1], label = 'simulated trajectory')

plt.scatter(y_pred[traj_idx, 0, 0], y_pred[traj_idx, 0, 1], color = 'green', label = 'decoded starting point')
plt.plot(y_pred[traj_idx, :, 0], y_pred[traj_idx, :, 1], label = 'decoded trajectory')
plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)
plt.legend(bbox_to_anchor=(1.5, 1.))

plt.savefig('trajectory2.png', bbox_inches='tight')
# plt.show()
