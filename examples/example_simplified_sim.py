import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR
np.random.seed(1)  # set seed
import os.path
from scipy.io import loadmat
from utils.plots import plot_predictive_1D, visualization_experiment
from src.model_fitting.gp_common import GPDataset, read_dataset
from config.configuration_parameters import ModelFitConfig as Conf
import matplotlib.pyplot as plt

# SETUP
y_dim = 9
train = False
x_cap = 16
n_bins = 40
hist_thresh = 1e-3

dataset_name = Conf.ds_name
save_dir = os.environ["SSGPR_PATH"] + "/data/" + dataset_name
os.makedirs(save_dir,exist_ok=True)

quad_sim_options = Conf.ds_metadata
df_train = read_dataset(dataset_name, True, quad_sim_options)
df_train = df_train.sample(frac=1).reset_index(drop=True)
gp_dataset = GPDataset(df_train, x_features=[y_dim], u_features=[], y_dim=y_dim,
                        cap=x_cap, n_bins=n_bins, thresh=hist_thresh, visualize_data=False)
gp_dataset.cluster(n_clusters=1, load_clusters=False, save_dir=save_dir, visualize_data=False)
X = gp_dataset.get_x(cluster=0)
Y = gp_dataset.get_y(cluster=0)

num_points = int(len(X)*0.7)

X_train = X[:num_points,:]
X_test = X[num_points:,:]
Y_train = Y[:num_points,:]
Y_test = Y[num_points:,:]

# Sorting (only for visualization purposes)
sorted_indices = np.argsort(X_train.flatten())
X_train = X_train[sorted_indices]
Y_train = Y_train[sorted_indices]

sorted_indices = np.argsort(X_test.flatten())
X_test = X_test[sorted_indices]
Y_test = Y_test[sorted_indices]

filename = 'ssgpr_model_' + str(y_dim) + '.pkl'
if (train):
    # create ssgpr instance
    nbf = 100  # number of basis functions
    ssgpr = SSGPR(num_basis_functions=nbf)
    ssgpr.add_data(X_train, Y_train, X_test, Y_test)
    ssgpr.optimize(restarts=3, verbose=True)
    ssgpr.save(os.path.join(save_dir,filename))
else:
    ssgpr = SSGPR.load(os.path.join(save_dir, filename))

# predict on the test points
mu, sigma, f_post = ssgpr.predict(X_test, sample_posterior=True)

# evaluate the performance
NMSE, MNLP = ssgpr.evaluate_performance()
print("Normalised mean squared error (NMSE): %.5f" % NMSE)
print("Mean negative log probability (MNLP): %.5f" % MNLP)

filename = 'ssgpr_model_' + str(y_dim) + '.png'
path = os.path.join(save_dir,filename)
plot_predictive_1D(path=path, X_train=X_train, Y_train=Y_train, Xs=X_test, mu=mu,
                   stddev=sigma, post_sample=f_post)
# SETUP
# y_dims = np.array([0])

visualization_experiment(quad_sim_options, dataset_name, x_cap=x_cap,hist_bins=n_bins,hist_thresh=hist_thresh,
                         x_vis_feats=[y_dim],u_vis_feats=[],y_vis_feats=y_dim,save_file_path=save_dir, ssgpr=ssgpr)