from trainer import Trainer
from trajectory_generator import TrajectoryGenerator
from utils.make_train_target import make_train_target

# generate training data
T = 20  # duration of simulated trajectories (seconds)
srate = 50  # sampling rate (Hz)

border_region = 0.03  # max. distance to wall (m)
sequence_length = T * srate  # number of steps in trajectory
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
mini_batch_size = 64
# n_data = mini_batch_size * 2
n_data = 100

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
position, velocity, head_dir = trajectory_generator.generate_trajectory()
train, target = make_train_target(position, velocity, head_dir)

# start training
trainer = Trainer(train, target, n_epochs=3)
trainer.train()