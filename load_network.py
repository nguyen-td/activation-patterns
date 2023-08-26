import torch
import pickle

from utils import make_train_data, create_ratemaps, TrajectoryGenerator

# load data
position, velocity, head_dir = torch.load('models/data.pt')
data = make_train_data(velocity.reshape(1, -1), head_dir.reshape(1, -1))

# load model
hidden_size = 256
mini_batch_size = 32
rnn_layer = 'custom'
rnn_model = torch.load(f'models/RNN-{hidden_size}-{rnn_layer}-model.pt')

rnn_model.eval()
aggregate_loss, y_pred, x = rnn_model.evaluate(data, position)

# spatial_maps = create_ratemaps(x, y_pred, box_width, box_height)
# with open("spatial_maps", "wb") as fp:   #Pickling
#     pickle.dump(spatial_maps, fp)