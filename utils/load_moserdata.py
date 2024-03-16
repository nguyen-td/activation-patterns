from scipy.io import loadmat
from pathlib import Path
import numpy as np


def load_moserdata(f_name, n_data):
    """
    Load grid cell data.

    Input:
        f_name: String
            Name of the saved MATLAB file, must include the .mat extension. The data is assumed to be in the 'data' folder.
        n_data: Scalar
            Number of trajectories to generate. Right now, only a single trajectory is used, i.e., the trajectory is copied [n_data] times.    
    Outputs:
        out_position: (n_data x T x 2) Numpy array
            Position in 2D (x and y coordinates)
        out_vel_x: (n_data x T) Numpy array
            Velocity (m/sample)
        out_vel_y: (n_data x T) Numpy array
            Head direction (rads)

    """

    # load data
    mat = loadmat(Path('data') / f_name)
    pos_x, pos_y, vel_x, vel_y = mat['save_mat'].T
    position = np.stack((pos_x[0][0:-1], pos_y[0][0:-1]))

    # create [n_data] trajectories (right now, by stacking them)
    out_position = np.full((n_data, position.shape[1], position.shape[0]), np.squeeze(position).T)
    out_vel_x = np.full((n_data, len(vel_x[0])), np.squeeze(vel_x[0]))
    out_vel_y = np.full((n_data, len(vel_y[0])), np.squeeze(vel_y[0]))

    return out_position, out_vel_x, out_vel_y