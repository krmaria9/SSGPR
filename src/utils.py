import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset
import os
import pandas as pd

def visualize_error(path, X_test, Y_test, Xs, mu, x_vis_feats, u_vis_feats):
    feats = x_vis_feats + u_vis_feats

    # Visualize error over features
    if X_test.shape[1]==10:
        _, axs_error = plt.subplots(2, 5, figsize=(18, 6))
        axs_error = axs_error.flatten()
    elif X_test.shape[1]==14:
        _, axs_error = plt.subplots(2, 7, figsize=(18, 6))
        axs_error = axs_error.flatten()
    elif X_test.shape[1]==30:
        _, axs_error = plt.subplots(5, 6, figsize=(18, 18))
        axs_error = axs_error.flatten()
    elif X_test.shape[1]==3:
        _, axs_error = plt.subplots(1, 3, figsize=(18, 6))
        axs_error = axs_error.flatten()
        # velocity_magnitudes = np.linalg.norm(X_test, axis=1)
        # X_test = np.column_stack((X_test, velocity_magnitudes))
    elif X_test.shape[1]==1:
        _, axs_error = plt.subplots(1, 1, figsize=(18, 18))
    else:
        return
    
    for idx, i in enumerate(range(X_test.shape[1])):
        if X_test.shape[1] > 1:
            # Scatter plot for training data error
            axs_error[idx].scatter(X_test[:, i], Y_test, c='blue', marker='o', alpha=0.7, label='Test Error')
            
            # Scatter plot for test data error
            if mu is not None:
                axs_error[idx].scatter(Xs[:, i], mu, c='red', marker='x', alpha=0.7, label='Predicted Error')
            axs_error[idx].grid(True)
        else:
            # Scatter plot for training data error
            axs_error.scatter(X_test[:, i], Y_test, c='blue', marker='o', alpha=0.7, label='Test Error')
            
            # Scatter plot for test data error
            if mu is not None:
                axs_error.scatter(Xs[:, i], mu, c='red', marker='x', alpha=0.7, label='Predicted Error')
            axs_error.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"{path}/error.png",dpi=300)
    plt.close()


def visualization_experiment(dataset_file,
                            x_cap, hist_bins, hist_thresh,
                            x_vis_feats, u_vis_feats, y_vis_feats,
                            save_file_path, ssgpr, alpha_all):
    
    dir_name = os.path.dirname(save_file_path)
    os.makedirs(dir_name, exist_ok=True)

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
    
    test_gp_ds = Dataset(test_ds, x_features=x_vis_feats, u_features=u_vis_feats, y_dim=y_vis_feats,
                           cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)

    # TODO (krmaria): need to fix for u
    X_init = test_gp_ds.get_x()
    Y_init = test_gp_ds.get_y()
    dt_test = test_gp_ds.get_dt()
    
    X_new = X_init
    y_test = Y_init
    t_vec = test_ds['timestamp']

    # #### EVALUATE GP ON TEST SET #### #
    mean_estimate_test, _, alpha_test = ssgpr.predict(Xs=X_new, Ys=y_test) # alpha specific from run
    mean_estimate_all = ssgpr.evaluate_prediction(Xs=X_new, alpha=alpha_all) # alpha from full dataset

    np.savetxt(f'{save_file_path}_alpha_test.csv', alpha_test, delimiter=",")

    # Error of nominal model
    nominal_diff = y_test

    # GP regresses model error, correct the predictions of the nominal model
    augmented_diff_test = nominal_diff - mean_estimate_test
    augmented_diff_all = nominal_diff - mean_estimate_all

    nominal_rmse = np.sqrt(np.mean(nominal_diff**2))
    augmented_rmse_test = np.sqrt(np.mean(augmented_diff_test**2))
    augmented_rmse_all = np.sqrt(np.mean(augmented_diff_all**2))

    labels = [r'$v_x$ error', r'$v_y$ error', r'$v_z$ error']

    plt.figure()
    for i in range(mean_estimate_test.shape[1]):
        plt.subplot(mean_estimate_test.shape[1], 1, i + 1)
        # plt.plot(t_vec, augmented_diff[:, i], label='out - pred_train')
        plt.plot(t_vec, augmented_diff_test[:, i], label='out - pred_test')
        plt.plot(t_vec, augmented_diff_all[:, i], label='out - pred_all')

        # # Fill area between -2*std and 2*std with lightgray color
        # lower_bound = augmented_diff[:, i] - std_estimate[:, i]
        # upper_bound = augmented_diff[:, i] + std_estimate[:, i]
        # plt.fill_between(t_vec, lower_bound, upper_bound, color='lightgray')

        if nominal_diff is not None:
            plt.plot(t_vec, nominal_diff[:, i], label='out')
            plt.title('Dt: %.2f s. Nom RMSE: %.3f. \nReduc test: %.2f%%. Reduc all: %.2f%%' % (
                float(np.mean(dt_test)), nominal_rmse,
                (1 - augmented_rmse_test / nominal_rmse) * 1e2, (1 - augmented_rmse_all / nominal_rmse) * 1e2))

        # plt.plot(t_vec, mean_estimate, 'g', label='pred_train')
        plt.plot(t_vec, mean_estimate_test, label='pred_test')
        plt.plot(t_vec, mean_estimate_all, label='pred_all')
        plt.ylabel(f'v_{y_vis_feats}')
        plt.legend()

        if i == mean_estimate_test.shape[1] - 1:
            plt.xlabel('time (s)')

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(f"{save_file_path}_pred.png", dpi=400)
    plt.close()
