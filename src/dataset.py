import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import glob
from src.model_fitting.gp_common import GPDataset

# Constants
NUM_BINS = 10
MIN_VEL = 3 # m/s
MAX_VEL = 20 # m/s

# Directory where CSV files are located
folder_path = os.environ["FLIGHTMARE_PATH"] + "/misc/data_sihao"

# Identify all CSV files in directory
all_files = [f for f in glob.glob(os.path.join(folder_path, "*.csv")) if "temp" not in os.path.basename(f)]

dataset_path = os.path.join(folder_path,'dataset')
os.makedirs(dataset_path, exist_ok=True)

# Lists to store relevant data
vel_mag = []
sample_data = []

for file in all_files:
    df = pd.read_csv(file)
    state_values = df['state_in'].str.strip('[]').str.split(',').apply(lambda x: [float(item) for item in x])
    file_vel_mag = state_values.apply(lambda x: np.sqrt(x[7]**2 + x[8]**2 + x[9]**2)).tolist()
    vel_mag.extend(file_vel_mag)  # Append velocity magnitudes to vel_mag
    sample_data.extend(df[['dt', 'state_in', 'input_in', 'state_out', 'state_pred']].values.tolist())

mask = [(MIN_VEL <= vmag <= MAX_VEL) for vmag in vel_mag]
vel_mag = [vmag for i, vmag in enumerate(vel_mag) if mask[i]]
sample_data = [data for i, data in enumerate(sample_data) if mask[i]]

# Sample M data points from each bin except the two lowest bins
counts, bin_edges = np.histogram(vel_mag, bins=NUM_BINS)
sorted_counts = sorted(counts)
M = sorted_counts[1]  # Since it's 0-indexed, 1 is the second position
samples = []

for i in range(NUM_BINS):
    mask = [bin_edges[i] <= vmag < bin_edges[i + 1] for vmag in vel_mag]
    bin_data = [data for j, data in enumerate(sample_data) if mask[j]]
    # If the size of the bin is larger than M, sample M random points. Otherwise, take all the points.
    samples.extend(random.sample(bin_data, min(M, len(bin_data))))

# Shuffle the samples
np.random.shuffle(samples)

# Store the sampled and shuffled data
columns = ['dt', 'state_in', 'input_in', 'state_out', 'state_pred']
sampled_df = pd.DataFrame(samples, columns=columns)
sampled_df.to_csv(os.path.join(dataset_path,'dataset_001.csv'), index=True)

# Histogram (compare before and after)
state_values = sampled_df['state_in'].str.strip('[]').str.split(',').apply(lambda x: [float(item) for item in x])
sample_vel_mags = state_values.apply(lambda x: np.sqrt(x[7]**2 + x[8]**2 + x[9]**2)).tolist()

plt.figure()
plt.hist(vel_mag, bins=NUM_BINS, alpha=0.5, label=f'All Data (N={len(vel_mag)})', color='blue')
plt.hist(sample_vel_mags, bins=NUM_BINS, alpha=0.5, label=f'Sampled Data (N={len(sample_vel_mags)})', color='red')
plt.xlabel('Velocity [m/s]')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{dataset_path}/sampled_histogram.png', dpi=400)
plt.close()

# Visualize dataset (3D)
x_feats = [7, 8, 9]
dataset = GPDataset(sampled_df, x_features=x_feats, u_features=[], y_dim=7,
                    cap=16, n_bins=40, thresh=1e-3, visualize_data=False)
dataset.cluster(n_clusters=1, load_clusters=False, visualize_data=False)

x = dataset.get_x(cluster=0)
y = dataset.get_y(cluster=0)

# TODO (krmaria): make this plot a function in gp visualization
combinations = [(0, 1), (0, 2), (1, 2)]  # Pairs of dimensions from x

y = abs(y) # Coloring based on abs value
mean_y = np.mean(y)
std_y = np.std(y)
vmin_val = mean_y - 2 * std_y
vmax_val = mean_y + 2 * std_y

fig, axs = plt.subplots(1, len(combinations), figsize=(15, 5))

for i, (dim1, dim2) in enumerate(combinations):
    sc = axs[i].scatter(x[:, dim1], x[:, dim2], c=y.squeeze(), s=1, cmap='viridis', vmin=vmin_val, vmax=vmax_val)  # c=y.squeeze() is the color scaling
    axs[i].set_xlabel(f'x_{dim1}')
    axs[i].set_ylabel(f'x_{dim2}')
    axs[i].grid(True)
    axs[i].set_ylim(-20,20)
    plt.colorbar(sc, ax=axs[i], label='y')  # Adding a colorbar for the y values

plt.tight_layout()
plt.savefig(f'{dataset_path}/combi.png', dpi=400)
plt.close()
