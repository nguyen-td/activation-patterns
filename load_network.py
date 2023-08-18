import torch
import pickle

from trajectory_generator import TrajectoryGenerator
from utils.make_train_data import make_train_data
from utils.create_ratemaps import create_ratemaps

# define parameters
T = 20  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
n_data_test = 32 * 200

# generate test data
trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data_test)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
test = make_train_data(velocity, head_dir)

# load model
hidden_size = 256
mini_batch_size = 32
rnn_layer = 'custom'
n_data = '30k'
rnn_model = torch.load(f'models/RNN-{hidden_size}-{rnn_layer}-{n_data}-model.pt')
# rnn_model = RNNModel(hidden_size, mini_batch_size, rnn_layer)
# rnn_model.double()
# rnn_model.load_state_dict(torch.load(f'models/RNN-{hidden_size}-{rnn_layer}-model.pt', map_location=torch.device('cpu')))

rnn_model.eval()
aggregate_loss, y_pred, x = rnn_model.evaluate(test, position)

spatial_maps = create_ratemaps(x, y_pred, box_width, box_height)
with open("spatial_maps", "wb") as fp:   #Pickling
    pickle.dump(spatial_maps, fp)