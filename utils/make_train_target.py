import numpy as np


def make_train_target(position, velocity, head_dir):
    """
    Make training and target (ground truth) data from simulated inputs, e.g., from simulated rat trajectories.

    Inputs:
        position: (batch_size x T x 2) Numpy array
            Position in 2D (x and y coordinates)
        velocity: (batch_size x T) Numpy array
            Velocity (m/sample)
        head_dir: (batch_size x T) Numpy array
            Head direction (rads)

    Output:
        train_data: (batch_size x 2 x T) Numpy array
            Training data
        target_data: (batch_size x 2 x T) Numpy array
            Target data
    """
    train_data = np.stack((velocity, head_dir), axis=1)
    target_data = np.transpose(position, (0, 2, 1))

    return train_data, target_data