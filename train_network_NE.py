import torch
print(torch.cuda.is_available())
import matplotlib.pyplot as plt
from pathlib import Path

from trainers.trainer_NE import Trainer_NE
from utils import make_train_data, TrajectoryGenerator, load_moserdata

# settings
mini_batch_size = 16
n_data = mini_batch_size * 1000
activation = 'relu' # type of activation function
data_type = 'sim' # "sim" or "real"
T = 1 # duration in minutes, for real data
lr = 1e-4 # default
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
device = "cpu:1"

# generate training data
if data_type == 'sim':
    T = 60  # duration of simulated trajectories (seconds)
    srate = 50  # sampling rate (Hz)

    border_region = 0.03  # max. distance to wall (m)
    sequence_length = T * srate  # number of steps in trajectory
    box_width = 2.2       # width of training environment (m)
    box_height = 2.2      # height of training environment (m)

    trajectory_generator = TrajectoryGenerator(sequence_length, border_region, box_width, box_height, n_data)
    position, vel_x, vel_y = trajectory_generator.generate_trajectory() # position, velocity, head_direction

    torch.save([position[0], vel_x[0], vel_y[0]], f'data/data-{activation}-{data_type}-{T}-NE.pt')
else:
    position, vel_x, vel_y = load_moserdata(f't2c1_{T}min.mat', n_data)
    torch.save([position[0], vel_x[0], vel_y[0]], f'data/data-{activation}-{data_type}-{T}-NE.pt')
    
train = make_train_data(vel_x, vel_y)

# # start training
pop_size = 200
hidden_size = 256
n_terations = 20000
rnn_layer = 'custom' 
model_name = f'RNN-{hidden_size}-{rnn_layer}-{activation}-{data_type}-{T}-NE'

trainer = Trainer_NE(device, train, position, model_name, hidden_size=hidden_size, mini_batch_size=mini_batch_size, pop_size=pop_size, num_actors="num_gpus", rnn_layer=rnn_layer, n_iterations=n_terations)
train_loss_epochs = trainer.train()

# plot training progress
plt.plot(train_loss_epochs)
plt.title('Training loss over epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss.png', bbox_inches='tight')

torch.save(train_loss_epochs, f'models/loss-{activation}-{data_type}-{T}-NE.pt')