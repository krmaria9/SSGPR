import argparse
import sys
import numpy as np
from model.ssgpr import SSGPR
import os.path
from utils.plots import plot_predictive_1D, visualization_experiment
from src.model_fitting.gp_common import GPDataset, read_dataset
from config.configuration_parameters import ModelFitConfig as Conf

def main(x_features, reg_y_dim, quad_sim_options, dataset_name, x_cap, hist_bins, hist_thresh, nbf, train, n_restarts):
    save_dir = os.environ["SSGPR_PATH"] + "/data/" + dataset_name
    os.makedirs(save_dir,exist_ok=True)
    
    df_train = read_dataset(dataset_name, True, quad_sim_options)
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    gp_dataset = GPDataset(df_train, x_features=[x_features], u_features=[], y_dim=reg_y_dim,
                        cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)
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

    filename = 'ssgpr_model_' + str(reg_y_dim)
    if (train):
        # create ssgpr instance
        ssgpr = SSGPR(num_basis_functions=nbf)
        ssgpr.add_data(X_train, Y_train, X_test, Y_test)
        ssgpr.optimize(restarts=n_restarts, verbose=True)
        ssgpr.save(os.path.join(save_dir,filename + '.pkl'))
    else:
        ssgpr = SSGPR.load(os.path.join(save_dir, filename + '.pkl'))

    # predict on the test points
    mu, sigma, f_post = ssgpr.predict(X_test, sample_posterior=True)

    # evaluate the performance
    NMSE, MNLP = ssgpr.evaluate_performance()
    print("Normalised mean squared error (NMSE): %.5f" % NMSE)
    print("Mean negative log probability (MNLP): %.5f" % MNLP)

    filename = 'ssgpr_model_' + str(reg_y_dim)
    path = os.path.join(save_dir, filename + '.png')
    plot_predictive_1D(path=path, X_train=X_train, Y_train=Y_train, Xs=X_test, mu=mu,
                    stddev=sigma, post_sample=f_post)

    visualization_experiment(quad_sim_options,dataset_name,x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,
                            x_vis_feats=[x_features],u_vis_feats=[],y_vis_feats=reg_y_dim,save_file_path=save_dir,ssgpr=ssgpr)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=int, default=1)

    parser.add_argument("--nbf", type=int, default="100",
                        help="Number of basis functions to use for the SSGP approximation")

    parser.add_argument('--x', type=int, default=7,
                        help='Regression X variables. Must be a list of integers between 0 and 12. Velocities xyz '
                             'correspond to indices 7, 8, 9.')

    parser.add_argument("--y", type=int, default=7,
                        help="Regression Y variable. Must be an integer between 0 and 12. Velocities xyz correspond to"
                             "indices 7, 8, 9.")

    args = parser.parse_args()
    
    # Use vx, vy, vz as input features
    x_feats = args.x
    
    # Regression dimension
    y_regressed_dim = args.y
    nbf = args.nbf
    train = args.train==1
    
    # Stuff from Conf
    hist_bins = Conf.histogram_bins
    hist_thresh = Conf.histogram_threshold
    x_cap = Conf.velocity_cap
    quad_sim_options = Conf.ds_metadata
    dataset_name = Conf.ds_name

    main(x_features=x_feats,reg_y_dim=y_regressed_dim,quad_sim_options=quad_sim_options,dataset_name=dataset_name,
         x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,nbf=nbf,train=train,n_restarts=3)
