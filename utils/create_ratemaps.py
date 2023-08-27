import numpy as np
from collections import Counter
import itertools
import matplotlib.pyplot as plt


def create_ratemaps(x, y_pred, box_width, box_height):
    """
    Compute spatial ratemaps by binning the agent's position into 2 cm x 2 cm bins, and computing the average firing rate within each bin. 

    Inputs:
        x: (T x n_hidden_neurons) Numpy array
            Activation of the units
        y_pred: (T x 2) Numpy array
            Predicted positions
        box_width: Scalar
            Width of the artificial box in meters
        box_heigtht: Scalar
            Height (in 2D, could also be length) of the artificial box in meters

    Output:
        spatial_maps: List of length n_neurons, each entry is a (T x 3) numpy array
            Data matrix containing the (mean) firing activity and the corresponding spatial positions in column 2 and 3 respectively (spatial_maps[:, 1:3])
    """

    spatial_maps = list()
    for neuron in range(x.shape[1]):
        # print(f"Neuron {neuron+1}\n................")

        # single neuron
        x_neuron = x[:, neuron]
        x_resh = x_neuron.reshape(-1, 1)
        y_resh = y_pred.reshape(-1, 2)

        # create bins
        n_bins = int(box_width * 100 / 2) # 2 cm bins
        H, xedges, yedges = np.histogram2d(y_resh[:, 0], y_resh[:, 1], bins=n_bins, range=[[-box_width / 2, box_height / 2], [-box_width / 2, box_height / 2]]) 
        bin_inds = np.digitize(y_resh, bins=yedges)
        y_bin = yedges[bin_inds - 1]
        data = np.hstack((x_resh, y_bin)) # create a single matrix for x and y

        # count the number of reappearing positions to average them
        coordinates = data[:, 1:3]
        dict_counts = Counter(map(tuple, coordinates))
        dupe_idx = [idx for idx, num in enumerate(list(dict_counts.values())) if num > 1]
        dupe_keys = np.array(list(dict_counts.keys()))[dupe_idx]

        means = np.zeros((len(dupe_keys), data.shape[1]))
        all_rows_combs = np.zeros(data.shape[0])
        for comb in range(len(dupe_keys)):
            rows_with_comb = np.all(coordinates == dupe_keys[comb], axis=1)
            all_rows_combs += rows_with_comb
            
            means[comb] = np.mean(data[rows_with_comb], axis=0)
            
        # only use mean activities in each bin
        data_new = np.delete(data, np.arange(0, data.shape[0], 1)[all_rows_combs.astype(bool)], 0)
        data_new = np.vstack((data_new, means))

        # save neuron activities
        spatial_maps.append(data_new)

    return spatial_maps

def plot_ratemaps(spatial_maps, hidden_size, box_width, box_height, resolution, smooth=True):
    """
    Plot spatial ratemaps and save the final figure.

    Inputs:
        spatial_maps: List of length n_neurons, each entry is a (T x 3) numpy array
            Data matrix containing the (mean) firing activity and the corresponding spatial positions in column 2 and 3 respectively (spatial_maps[:, 1:3])
        hidden_size: Scalar
            Number of hidden neurons in the recurrent layer
        box_width: Scalar
            Width of the artificial box in meters
        box_heigtht: Scalar
            Height (in 2D, could also be length) of the artificial box in meters
        resolution: Scalar
            Size of each bin
        smooth: Boolean
            If True, Gaussian smoothing will be applied
    """
    
    # get all combinations
    p = itertools.product(np.arange(0, np.sqrt(hidden_size), 1), np.arange(0, np.sqrt(hidden_size), 1))
    min_firing = np.min(spatial_maps)
    max_firing = np.max(spatial_maps)

    fig, axs = plt.subplots(int(np.sqrt(hidden_size)), int(np.sqrt(hidden_size)), figsize=(15, 15))
    for idx, comb in enumerate(p):
        x_plt = spatial_maps[idx][:, 0]
        y_plt = spatial_maps[idx][:, 1]
        z_plt = spatial_maps[idx][:, 2]

        x_bins = y_bins = np.arange(-box_width / 2, box_height / 2, resolution)  # 110 bins of size 2 cm

        heatmap, _, _ = np.histogram2d(x_plt, y_plt, bins=[x_bins, y_bins], weights=z_plt)
        
        if smooth:
            axs[int(comb[0]), int(comb[1])].imshow(heatmap.T, extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()], vmin=min_firing, vmax=max_firing, origin='lower', aspect='auto', interpolation='gaussian', cmap='jet')
        else:
            axs[int(comb[0]), int(comb[1])].imshow(heatmap.T, extent=[x_bins.min(), x_bins.max(), y_bins.min(), y_bins.max()], vmin=min_firing, vmax=max_firing, origin='lower', aspect='auto', cmap='jet')
        axs[int(comb[0]), int(comb[1])].set_xlim([-box_width / 2, box_width / 2])
        axs[int(comb[0]), int(comb[1])].set_ylim([-box_width / 2, box_width / 2])
        axs[int(comb[0]), int(comb[1])].set_axis_off()

    fig.savefig('ratemaps.png', bbox_inches='tight')


