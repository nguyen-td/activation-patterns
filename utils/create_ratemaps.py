import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt


def create_ratemaps(u, y_pred, box_width, box_height, res=50):
    """
    Compute spatial ratemaps by binning the agent's position into 2 cm x 2 cm bins, and computing the average firing rate within each bin. 

    Inputs:
        u: (T x n_hidden_neurons) Numpy array
            Activation of the units
        y_pred: (T x 2) Numpy array
            Predicted positions
        box_width: Scalar
            Width of the artificial box in meters
        box_heigtht: Scalar
            Height (in 2D, could also be length) of the artificial box in meters
        res: Scalar
            Heatmap resolution

    Output:
        activations: (n_hidden_neurons x res x res) Numpy array
           Data array containing the firing heatmap of each neuron
        x_pos: (1000, ) Numpy array
            Values of the x-coordinate, adjusted for the specificed resolution
        y_pos: (1000, ) Numpy array
            Values of the y-coordinate, adjusted for the specificed resolution
    """

    T = u.shape[0]
    n_neurons = u.shape[1]
    activations = np.zeros([n_neurons, res, res]) 
    counts = np.zeros([res, res])

    x_pos = (y_pred[:, 0] + box_width/2) / (box_width) * res
    y_pos = (y_pred[:, 1] + box_height/2) / (box_height) * res

    for t in range(T):
        x = x_pos[t]
        y = y_pos[t]
        if x >=0 and x < res and y >=0 and y < res:
            counts[int(x), int(y)] += 1
            activations[:, int(x), int(y)] += u[t, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]

    return activations, x_pos, y_pos

def plot_single_ratemap():
    pass

def plot_ratemaps(activations, x_pos, y_pos, fname, smooth=True):
    """
    Plot spatial ratemaps and save the final figure.

    Inputs:
        activations: (n_hidden_neurons x res x res) Numpy array
           Data array containing the firing heatmap of each neuron
        x_pos: (1000, ) Numpy array
            Values of the x-coordinate, adjusted for the specificed resolution
        y_pos: (1000, ) Numpy array
            Values of the y-coordinate, adjusted for the specificed resolution
        fname: String
            File name
        smooth: Boolean
            If True, Gaussian smoothing will be applied
    """
    
    hidden_size = activations.shape[0]

    # get all combinations
    p = itertools.product(np.arange(0, np.sqrt(hidden_size), 1), np.arange(0, np.sqrt(hidden_size), 1))
    min_firing = np.min(activations)
    max_firing = np.max(activations)

    fig, axs = plt.subplots(int(np.sqrt(hidden_size)), int(np.sqrt(hidden_size)), figsize=(15, 15))
    for idx, comb in enumerate(p):
        if smooth:
            axs[int(comb[0]), int(comb[1])].imshow(activations[idx], extent=[x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()], vmin=min_firing, vmax=max_firing, origin='lower', aspect='auto', interpolation='gaussian', cmap='jet')
        else:
            axs[int(comb[0]), int(comb[1])].imshow(activations[idx], extent=[x_pos.min(), x_pos.max(), y_pos.min(), y_pos.max()], vmin=min_firing, vmax=max_firing, origin='lower', aspect='auto', cmap='jet')
        axs[int(comb[0]), int(comb[1])].set_axis_off()


    fig.savefig(f'ratemaps\{fname}.png', bbox_inches='tight')


