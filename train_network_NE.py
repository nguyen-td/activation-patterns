import torch
print(torch.cuda.is_available())
from pathlib import Path

from trainers.trainer_NE import Trainer_NE
from utils import make_train_data, TrajectoryGenerator

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
torch.save([position[0], velocity[0], head_dir[0]], 'data-NE.pt')
train = make_train_data(velocity, head_dir)

# # start training
pop_size = 200
hidden_size = 256
n_terations = 500
rnn_layer = 'native' 
model_name = f'RNN-{hidden_size}-{rnn_layer}-NE'

trainer = Trainer_NE(train, position, model_name, hidden_size=hidden_size, mini_batch_size=mini_batch_size, pop_size=pop_size, num_actors="num_gpus", rnn_layer=rnn_layer, n_iterations=n_terations)
trainer.train()