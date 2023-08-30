import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from src.model_fitting.gp_common import GPDataset, read_dataset
import os
import casadi as ca
import pandas as pd

def plot_convergence(path, convergence):
    N = convergence.shape[0]
    X = np.arange(N)
    Y = convergence[:,0]
    plt.figure()
    plt.plot(X, Y)
    plt.grid()
    plt.ylabel("Negative marginal log likelihood")
    plt.xlabel("Number of iterations")
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.show()

def plot_predictive_1D(path=None, X_train=None, Y_train=None, Xs=None, mu=None, stddev=None, post_sample=None):
    """
    Plot the predictive distribution for one dimensional data.

    See example_1D.py for use.
    
    Parameters
    ----------
    path : str
        Path to save figure. If no path is provided then the figure is not saved.
        
    X_train : numpy array of shape (N, 1)
        Training data.
        
    Y_train : numpy array of shape (N, 1)
        Training targets.
        
    Xs : numpy array of shape (n, 1)
        New points used to predict on.
        
    mu : numpy array of shape (n, 1)
        Predictive mean generated from new points Xs.
        
    stddev : numpy array of shape (n, 1)
        Standard deviation generated from the new points Xs.
        
    post_sample : numpy array of shape (n, num_samples)
        Samples from the posterior distribution over the model parameters. 
    """
    plt.figure()
    if (X_train is not None) and (Y_train is not None):
        plt.plot(X_train, Y_train, '*', label="Training data", color='blue')  # training data
    if post_sample is not None:
        plt.plot(Xs, post_sample, '--', label="Posterior sample")
    plt.plot(Xs, mu, 'k', lw=2, label="Predictive mean")
    plt.fill_between(Xs.flat, (mu - 2 * stddev).flat, (mu + 2 * stddev).flat, color="#dddddd",
                     label="95% confidence interval")
    plt.grid()
    # plt.autoscale(enable=True, axis='x', tight=True)
    plt.xlabel("inputs, X")
    plt.ylabel("targets, Y")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    # save img
    # if path is not None:
    plt.savefig(path, dpi=300)
    plt.close()
    # plt.show()

def plot_predictive_2D(path=None, X_train=None, Y_train=None, Xs1=None, Xs2=None, mu=None, stddev=None):
    """
    Plot the predictive distribution for one dimensional data.

    See example_2D.py for use.

    Parameters
    ----------
    path : str
        Path to save figure. If no path is provided then the figure is not saved.

    X_train : numpy array of shape (N, 1)
        Training data.

    Y_train : numpy array of shape (N, 1)
        Training targets.

    Xs1 : numpy array of shape (n, n)
        New points used to predict on. Xs1 should be generated with np.meshgrid (see example_2D.py).

    Xs2 : numpy array of shape (n, n)
        New points used to predict on. Xs2 should be generated with np.meshgrid (see example_2D.py).

    mu : numpy array of shape (n, n)
        Predictive mean generated from new points Xs1 and Xs2.

    stddev : numpy array of shape (n, n)
        Standard deviation generated from the new points Xs1 and Xs2.

    post_sample : numpy array of shape (n, num_samples)
        Samples from the posterior distribution over the model parameters. 
    """
    # instantiate figure
    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    # plot scatter of training data.
    if (X_train is not None) and (Y_train is not None):
        ax.scatter(X_train[:,0], X_train[:,1], Y_train, label="Training data", c='red')
    # plot lower 95% confidence interval
    if stddev is not None:
        p1 = ax.plot_surface(Xs1, Xs2, mu-2*stddev, label="Lower 95% confidence interval",
                            color='yellow' ,alpha=0.2, linewidth=0, antialiased=False)
        p1._facecolors2d = p1._facecolor3d # fixes matplotlib bug in surface legend
        p1._edgecolors2d = p1._edgecolor3d
    # plot surface of predicted data
    if (Xs1 is not None) and (Xs2 is not None) and (mu is not None):
        p2 = ax.plot_surface(Xs1, Xs2, mu, label="Predictive mean",
                             cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(p2, shrink=0.5, aspect=5) # Add a color bar which maps values to colors.
        p2._facecolors2d = p2._facecolor3d
        p2._edgecolors2d = p2._edgecolor3d
    # plot upper 95% confidence interval
    if stddev is not None:
        p3 = ax.plot_surface(Xs1, Xs2, mu+2*stddev, label="Upper 95% confidence interval",
                             color='green', alpha=0.2, linewidth=0, antialiased=False)
        p3._facecolors2d = p3._facecolor3d
        p3._edgecolors2d = p3._edgecolor3d
    # axis labels
    ax.set_xlabel("inputs, X1")
    ax.set_ylabel("inputs, X2")
    ax.set_zlabel("targets, y")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='best')
    ax.grid(True)
    plt.tight_layout()
    # save img
    # if path is not None:
    plt.savefig(path, dpi=300)
    plt.close()
    # plt.show()

def visualize_data(path, X_train, Y_train, Xs, mu, stddev):
    feature_combinations = [(0, 1), (0, 2), (1, 2)]
    
    # Get the global color limits for consistent color mapping
    vmin = min(np.min(Y_train), np.min(mu))
    vmax = max(np.max(Y_train), np.max(mu))
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (i, j) in enumerate(feature_combinations):
        # Scatter plot for training data
        sc1 = axs[idx].scatter(X_train[:, i], X_train[:, j], c=Y_train.ravel(), cmap='viridis', vmin=vmin, vmax=vmax, marker='o', alpha=0.7, label='Training Data')
        
        # Scatter plot for predictions using mu values
        sc2 = axs[idx].scatter(Xs[:, i], Xs[:, j], c=mu.ravel(), cmap='viridis', vmin=vmin, vmax=vmax, marker='x', alpha=0.7, label='Predictions')

        axs[idx].set_xlabel(f'Feature {i}')
        axs[idx].set_ylabel(f'Feature {j}')
        axs[idx].legend()
        axs[idx].grid(True)

    cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.02]) # defines the position and size of the colorbar
    fig.colorbar(sc1, cax=cbar_ax, orientation='horizontal')
    plt.suptitle("\n")
    plt.tight_layout()
    plt.savefig(f"{path}_comb.png")
    plt.close()

    # Visualize error over features
    fig_error, axs_error = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, i in enumerate(range(3)):
        # Scatter plot for training data error
        axs_error[idx].scatter(X_train[:, i], Y_train, c='blue', marker='o', alpha=0.7, label='Training Data Error')
        
        # Scatter plot for test data error
        axs_error[idx].scatter(Xs[:, i], mu, c='red', marker='x', alpha=0.7, label='Predicted Error')

        axs_error[idx].set_xlabel(f'Feature {i}')
        axs_error[idx].set_ylabel('Error')
        axs_error[idx].legend()
        axs_error[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{path}_error.png")
    plt.close()

# Adapted from gp_visualization (ros_gp_mpc)
def visualization_experiment(dataset_file,
                            x_cap, hist_bins, hist_thresh,
                            x_vis_feats, u_vis_feats, y_vis_feats,
                            save_file_path, ssgpr):

    # # #### GP ENSEMBLE LOADING #### #
    # if pre_set_gp is None:
    #     load_options = {"git": load_model_version, "model_name": load_model_name, "params": quad_sim_options}
    #     loaded_models = load_pickled_models(model_options=load_options)
    #     if loaded_models is None:
    #         raise FileNotFoundError("Model not found")
    #     gp_ensemble = restore_gp_regressors(loaded_models)
    # else:
    #     gp_ensemble = pre_set_gp

    # #### DATASET LOADING #### #
    # Pre-set labels of the data:
    labels_x = [
        r'$p_x\:\left[m\right]$', r'$p_y\:\left[m\right]$', r'$p_z\:\left[m\right]$',
        r'$q_w\:\left[rad\right]$', r'$q_x\:\left[rad\right]$', r'$q_y\:\left[rad\right]$', r'$q_z\:\left[rad\right]$',
        r'$v_x\:\left[\frac{m}{s}\right]$', r'$v_y\:\left[\frac{m}{s}\right]$', r'$v_z\:\left[\frac{m}{s}\right]$',
        r'$w_x\:\left[\frac{rad}{s}\right]$', r'$w_y\:\left[\frac{rad}{s}\right]$', r'$w_z\:\left[\frac{rad}{s}\right]$'
    ]
    labels_u = [r'$u_1$', r'$u_2$', r'$u_3$', r'$u_4$']
    labels = [labels_x[feat] for feat in x_vis_feats]
    labels_ = [labels_u[feat] for feat in u_vis_feats]
    labels = labels + labels_

    test_ds = pd.read_csv(dataset_file)
    test_gp_ds = GPDataset(test_ds, x_features=x_vis_feats, u_features=u_vis_feats, y_dim=y_vis_feats,
                           cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)

    # if isinstance(dataset_name, str):
    #     test_ds = read_dataset(dataset_name, True, quad_sim_options)
    #     test_gp_ds = GPDataset(test_ds, x_features=x_vis_feats, u_features=u_vis_feats, y_dim=y_vis_feats,
    #                             cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)
    # else:
    #     test_gp_ds = dataset_name

    x_test = test_gp_ds.get_x(pruned=True, raw=True)
    y_test = test_gp_ds.get_y(pruned=True, raw=False)
    dt_test = test_gp_ds.get_dt(pruned=True)
    x_pred = test_gp_ds.get_x_pred(pruned=True, raw=False)

    # #### EVALUATE GP ON TEST SET #### #
    mean_estimate, std_estimate = ssgpr.predict(x_test[:,x_vis_feats])
    
    # Predict using symbolic method
    f_mu, f_stddev = ssgpr.predict_symbolic(ca.SX.sym('X', x_test[:,x_vis_feats].shape[1]).T, type_function=True)
    res_mu = [f_mu(row) for row in x_test[:, x_vis_feats]]
    f_mu_val = np.array([float(val) for val in res_mu]).reshape(-1,1)
    res_std = [f_stddev(row) for row in x_test[:, x_vis_feats]]
    f_std_val = np.array([float(val) for val in res_std]).reshape(-1,1)
    
    # tol = 1e-6
    # print('####### MU #######')
    # diff_indices = np.where(np.abs(f_mu_val - mu) > tol)
    # for i, j in zip(*diff_indices):
    #     print(f"Element at ({i}, {j}): f_sym={f_mu_val[i, j]}, num={mu[i, j]}")

    # print('####### STDDEV #######')
    # diff_indices = np.where(np.abs(f_std_val - stddev) > tol)
    # for i, j in zip(*diff_indices):
    #     print(f"Element at ({i}, {j}): f_sym={f_std_val[i, j]}, num={stddev[i, j]}")

    mean_estimate *= dt_test[:, np.newaxis]
    std_estimate *= dt_test[:, np.newaxis]

    # Undo dt normalization
    y_test *= dt_test[:, np.newaxis]

    # Error of nominal model
    nominal_diff = y_test

    # GP regresses model error, correct the predictions of the nominal model
    augmented_diff = nominal_diff - mean_estimate

    nominal_rmse = np.sqrt(np.mean(nominal_diff**2))
    augmented_rmse = np.sqrt(np.mean(augmented_diff**2))

    # TODO (krmaria): Torrente was using this
    # nominal_mae = np.mean(np.abs(nominal_diff), 0)
    # augmented_mae = np.mean(np.abs(augmented_diff), 0)

    labels = [r'$v_x$ error', r'$v_y$ error', r'$v_z$ error']
    t_vec = np.cumsum(dt_test)
    plt.figure()
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    for i in range(std_estimate.shape[1]):
        plt.subplot(std_estimate.shape[1], 1, i+1)
        plt.plot(t_vec, np.zeros(augmented_diff[:, i].shape), 'k')
        plt.plot(t_vec, augmented_diff[:, i], 'b', label='augmented_err')
        # plt.plot(t_vec, augmented_diff[:, i] - 3 * std_estimate[:, i], ':c')
        # plt.plot(t_vec, augmented_diff[:, i] + 3 * std_estimate[:, i], ':c', label='3 std')
        if nominal_diff is not None:
            plt.plot(t_vec, nominal_diff[:, i], 'r', label='nominal_err')
            plt.title('Mean dt: %.2f s. Nom RMSE: %.5f [m/s].  Aug RMSE: %.5f [m/s]' % (
                float(np.mean(dt_test)), nominal_rmse, augmented_rmse))
        else:
            plt.title('Mean dt: %.2f s. Aug RMSE: %.5f [m/s]' % (
                float(np.mean(dt_test)), float(augmented_rmse)))

        plt.plot(t_vec, mean_estimate, 'g', label='predicted_err')
        plt.ylabel(f'v_{y_vis_feats}')
        plt.legend()

        if i == std_estimate.shape[1] - 1:
            plt.xlabel('time (s)')

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{save_file_path}_pred.png", dpi=400)
    plt.close()