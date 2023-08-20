import sys
sys.path.append("..")
import numpy as np
from model.ssgpr import SSGPR
from utils.plots import plot_predictive_1D
from math import floor
np.random.seed(1)  # set seed
import os

##################################################
#           Example for 1D data                  #
##################################################

# load the data
data_dir = os.environ["SSGPR_PATH"] + "/data/"
data = np.load(data_dir + "example_data/data_1D.npy")
n = floor(0.8 * data.shape[0])
X_train = data[:n,0].reshape(-1,1)
Y_train = data[:n,1].reshape(-1,1)
X_test = data[n:,0].reshape(-1,1)
Y_test = data[n:,1].reshape(-1,1)

# create ssgpr instance
nbf = 100  # number of basis functions
# ssgpr = SSGPR(num_basis_functions=nbf)
# ssgpr.add_data(X_train, Y_train, X_test, Y_test)
# ssgpr.optimize(restarts=3, verbose=True)
# ssgpr.save(os.path.join(data_dir,'example_data/example_1D.pkl'))
ssgpr = SSGPR.load(os.path.join(data_dir,'example_data/example_1D.pkl'))

# create some prediction points
Xs = np.linspace(-10,10,100).reshape(-1,1)
mu, sigma, f_post = ssgpr.predict(Xs, sample_posterior=True, num_samples=3)
NMSE, MNLP = ssgpr.evaluate_performance(restarts=1)
print("Normalised mean squared error (NMSE): %.5f" % NMSE)
print("Mean negative log probability (MNLP): %.5f" % MNLP)

path = data_dir + "example_data/example_1D.png"
# plot results
plot_predictive_1D(path=path, X_train=X_train, Y_train=Y_train, Xs=Xs, mu=mu,
                   stddev=sigma, post_sample=f_post)