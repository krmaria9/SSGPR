import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from src.model_fitting.gp_common import GPDataset, read_dataset
import os
import casadi as ca
import pandas as pd
from src.utils.utils import undo_jsonify
from src.model_fitting.gp_common import world_to_body_velocity_mapping
from matplotlib.ticker import AutoMinorLocator

REMOVE_IDX = -10

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
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(path, dpi=300)
    plt.close()

def compute_rmse(error_matrix):
    return np.sqrt(np.mean(error_matrix**2, axis=1))

def plot_rmse_vs_feature(rmse, data, labels, path):
    n_subplots = data.shape[1]
    fig, axs = plt.subplots(n_subplots, 1, figsize=(10, n_subplots * 5))
    for i in range(n_subplots):
        axs[i].scatter(data[:, i], rmse, alpha=0.5)
        axs[i].set_xlabel(labels[i])
        axs[i].set_ylabel('RMSE')
        axs[i].grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def create_steps_ahead_matrix(X, n_steps_ahead, n_pred_samples):
    """
    For a given matrix X, this function creates a new matrix by concatenating rows that are N steps ahead,
    for M times.

    :param X: The input matrix
    :param N: Steps ahead
    :param M: Number of times to concatenate
    :return: New matrix
    """
    parts = [X[i:-n_steps_ahead*n_pred_samples+i] for i in range(0, n_steps_ahead*n_pred_samples, n_steps_ahead)]
    return np.hstack(parts)

def visualize_prediction_all(dataset_files, results, save_file_path):
    # Extracting values for plotting
    nominal_rmses = [result[0] for result in results]
    augmented_rmses = [result[1] for result in results]
    reductions = [result[2] for result in results]
    
    # Plotting
    x = list(range(len(dataset_files)))
    plt.figure(figsize=(15, 6))

    # Set a width for each bar
    bar_width = 0.25

    # Positions of bars
    r1 = [i - bar_width for i in x]
    r2 = x
    r3 = [i + bar_width for i in x]

    # Nominal RMSE Bar
    nom_mean = np.mean(nominal_rmses)
    plt.bar(r1, nominal_rmses, width=bar_width, label=f'Nominal RMSE = {nom_mean:.2f}', align='center')
    plt.axhline(nom_mean, color='blue', linestyle='dashed', linewidth=1)

    # Augmented RMSE Bar
    aug_mean = np.mean(augmented_rmses)
    plt.bar(r2, augmented_rmses, width=bar_width, label=f'Augmented RMSE = {aug_mean:.2f}', align='center')
    plt.axhline(aug_mean, color='orange', linestyle='dashed', linewidth=1)

    # Reduction Bar
    reduc_mean = np.mean(reductions)
    plt.bar(r3, reductions, width=bar_width, label=f'Augmented/Nominal = {reduc_mean:.2f}', align='center', color='green')
    plt.axhline(reduc_mean, color='green', linestyle='dashed', linewidth=1,)

    ax = plt.gca()  # Get the current Axes instance
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    plt.xticks(x, [os.path.basename(f).replace("_states_inputs_out.csv", "") for f in dataset_files], rotation=45, ha="right")
    plt.ylabel('RMSE')
    plt.legend()
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5) 
    plt.savefig(f"{save_file_path}_pred_all.png", dpi=400)
    plt.close()

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

def visualize_error(path, X_test, Y_test, x_vis_feats, u_vis_feats, z_vis_feats):
    feats = x_vis_feats + u_vis_feats

    # Visualize error over features
    if X_test.shape[1]==10:
        _, axs_error = plt.subplots(2, 5, figsize=(18, 6))
        axs_error = axs_error.flatten()
    else:
        _, axs_error = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, i in enumerate(range(X_test.shape[1])):
        # Scatter plot for training data error
        axs_error[idx].scatter(X_test[:, i], Y_test, c='blue', marker='o', alpha=0.7, label='Test Error')
        
        # Scatter plot for test data error
        # axs_error[idx].scatter(Xs[:, i], mu, c='red', marker='x', alpha=0.7, label='Predicted Error')

        if i < len(x_vis_feats):
            axs_error[idx].set_xlabel(f'x_{feats[i]}')
        else:
            axs_error[idx].set_xlabel(f'u_{feats[i]}')
        axs_error[idx].set_ylabel('Error')
        axs_error[idx].legend()
        axs_error[idx].grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{path}_error.png")
    plt.close()

# Adapted from gp_visualization (ros_gp_mpc)
def visualization_experiment(dataset_file,
                            x_cap, hist_bins, hist_thresh,
                            x_vis_feats, u_vis_feats, z_vis_feats, y_vis_feats,
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
    labels_u = [r'$u_1$', r'$u_2$', r'$u_3$', r'$u_4$', r'$u_5$', r'$u_6$', r'$u_7$', r'$u_8$']
    labels = [labels_x[feat] for feat in x_vis_feats]
    labels_ = [labels_u[feat] for feat in u_vis_feats]
    labels = labels + labels_

    test_ds = pd.read_csv(dataset_file)
    
    original_length = len(test_ds) 
    test_ds = test_ds.drop_duplicates(subset=['state_in', 'input_in'], keep='first')
    final_length = len(test_ds)
    print(f"{original_length-final_length} rows were dropped out of {original_length}")
    
    test_ds = test_ds[:REMOVE_IDX]
    
    test_gp_ds = GPDataset(test_ds, x_features=x_vis_feats, u_features=u_vis_feats, y_dim=y_vis_feats,
                           cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)

    test_gp_ds.cluster(n_clusters=1, load_clusters=False, visualize_data=False)

    # TODO (krmaria): need to fix for u
    X_init = test_gp_ds.get_x(cluster=0)
    Y_init = test_gp_ds.get_y(cluster=0)
    dt_test = test_gp_ds.get_dt(pruned=True)
    x_pred = test_gp_ds.get_x_pred(pruned=True, raw=False)
    
    # X = np.concatenate((x_test[:,x_vis_feats], u_test[:,u_vis_feats], z_test[:,z_vis_feats]), axis=1)

    # X_new = create_steps_ahead_matrix(X_init,n_steps_ahead,n_pred_samples)
    # y_test = Y_init[n_steps_ahead*n_pred_samples:]
    
    # t_vec = test_ds['timestamp'][n_steps_ahead*n_pred_samples:]
    # dt_test = dt_test[n_steps_ahead*n_pred_samples:]
    
    X_new = X_init
    y_test = Y_init
    t_vec = test_ds['timestamp']

    # #### EVALUATE GP ON TEST SET #### #
    mean_estimate, std_estimate, _, alpha = ssgpr.predict(X_new, sample_posterior=True)
    # mean_estimate, std_estimate, _, alpha = ssgpr.predict(np.concatenate((x_test[:,x_vis_feats], u_test[:,u_vis_feats], z_test[:,z_vis_feats]), axis=1), sample_posterior=True)
    # mean_estimate, std_estimate, _, alpha = ssgpr.predict(np.concatenate((x_test[:,x_vis_feats], u_test[:,u_vis_feats]), axis=1), sample_posterior=True)

    mean_estimate *= dt_test[:, np.newaxis]
    std_estimate *= dt_test[:, np.newaxis]
    # mean_estimate_2 *= dt_test[:, np.newaxis]

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
    # t_vec = np.cumsum(dt_test)
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
            plt.title('Dt: %.2f s. Nom RMSE: %.3f.  Aug RMSE: %.3f. Reduc: %.2f' % (
                float(np.mean(dt_test)), nominal_rmse, augmented_rmse, augmented_rmse/nominal_rmse))
        else:
            plt.title('Dt: %.2f s. Aug RMSE: %.5f [m/s]' % (
                float(np.mean(dt_test)), float(augmented_rmse)))

        plt.plot(t_vec, mean_estimate, 'g', label='predicted_err')
        # plt.plot(t_vec, mean_estimate_2, '--y', label='predicted_err_2')
        plt.ylabel(f'v_{y_vis_feats}')
        plt.legend()

        if i == std_estimate.shape[1] - 1:
            plt.xlabel('time (s)')

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{save_file_path}_pred.png", dpi=400)
    plt.close()
    
def analyze_experiment(dataset_files,
                       x_cap, hist_bins, hist_thresh,
                       x_vis_feats, u_vis_feats, z_vis_feats, y_vis_feats,
                       save_file_path, ssgpr):
    labels_x = [
        r'$p_x\:\left[m\right]$', r'$p_y\:\left[m\right]$', r'$p_z\:\left[m\right]$',
        r'$q_w\:\left[rad\right]$', r'$q_x\:\left[rad\right]$', r'$q_y\:\left[rad\right]$', r'$q_z\:\left[rad\right]$',
        r'$v_x\:\left[\frac{m}{s}\right]$', r'$v_y\:\left[\frac{m}{s}\right]$', r'$v_z\:\left[\frac{m}{s}\right]$',
        r'$w_x\:\left[\frac{rad}{s}\right]$', r'$w_y\:\left[\frac{rad}{s}\right]$', r'$w_z\:\left[\frac{rad}{s}\right]$'
    ]
    labels_u = [r'$u_1$', r'$u_2$', r'$u_3$', r'$u_4$', r'$u_5$', r'$u_6$', r'$u_7$', r'$u_8$']

    all_rmse = []
    all_u_test = []
    all_x_test = []
    results = []

    for dataset_file in dataset_files:
        test_ds = pd.read_csv(dataset_file)
        
        test_gp_ds = GPDataset(test_ds, x_features=x_vis_feats, u_features=u_vis_feats, y_dim=y_vis_feats,
                               cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)

        x_test = test_gp_ds.get_x(pruned=True, raw=True)
        u_test = test_gp_ds.get_u(pruned=True, raw=True)
        z_test = test_gp_ds.get_z(pruned=True, raw=True)
        y_test = test_gp_ds.get_y(pruned=True, raw=False)
        dt_test = test_gp_ds.get_dt(pruned=True)

        mean_estimate, _, _, alpha = ssgpr.predict(np.concatenate((x_test[:,x_vis_feats], u_test[:,u_vis_feats], z_test[:,z_vis_feats]), axis=1), sample_posterior=True)

        mean_estimate *= dt_test[:, np.newaxis]
        y_test *= dt_test[:, np.newaxis]

        nominal_diff = y_test
        error_matrix = mean_estimate - nominal_diff
        rmse_per_sample = compute_rmse(error_matrix)
        
        all_rmse.extend(rmse_per_sample)
        all_u_test.extend(u_test[:,u_vis_feats])
        all_x_test.extend(x_test[:,x_vis_feats])

        augmented_diff = nominal_diff - mean_estimate
        nominal_rmse = np.sqrt(np.mean(nominal_diff**2))
        augmented_rmse = np.sqrt(np.mean(augmented_diff**2))
        reduction = augmented_rmse/nominal_rmse
        results.append((nominal_rmse, augmented_rmse, reduction))

    # Convert lists to numpy arrays
    all_rmse = np.array(all_rmse)
    all_u_test = np.array(all_u_test)
    all_x_test = np.array(all_x_test)
    
    # # RMSE vs Input feature
    # if u_vis_feats:
    #     plot_rmse_vs_feature(all_rmse, all_u_test, [labels_u[feat] for feat in u_vis_feats], path=f"{save_file_path}_rmse_u.png")

    # if x_vis_feats:
    #     plot_rmse_vs_feature(all_rmse, all_x_test, [labels_x[feat] for feat in x_vis_feats], path=f"{save_file_path}_rmse_x.png")

    # Experiment Plot
    visualize_prediction_all(dataset_files, results, save_file_path)