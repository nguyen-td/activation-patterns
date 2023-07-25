from trajectory_generator import TrajectoryGenerator
from torch.utils.data import DataLoader


sequence_length = 20  # number of steps in trajectory
border_region = 0.03  # max. distance to the wall (m)
box_width = 2.2       # width of training environment (m)
box_height = 2.2      # height of training environment (m)
batch_size = 10

trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, batch_size)
position, velocity, head_dir = trajectory_generator.generate_trajectory()

print(position.shape)
print(velocity.shape)
print(head_dir.shape)



