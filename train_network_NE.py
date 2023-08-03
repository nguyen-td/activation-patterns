import torch
print(torch.cuda.is_available())

from trainers.trainer_NE import Trainer_NE
from trajectory_generator import TrajectoryGenerator
from utils import make_train_data

# generate training data
T = 5  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 32
n_data = mini_batch_size * 2

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
train = make_train_data(velocity, head_dir)

# # start training
pop_size = 10
hidden_size = 100
rnn_layer = 'custom' 
model_name = f'RNN-{hidden_size}-{rnn_layer}-NE'

trainer = Trainer_NE(train, position, model_name, hidden_size=hidden_size, pop_size=pop_size, num_actors=2, rnn_layer=rnn_layer, n_epochs=5)
trainer.train()