import argparse
import sys
import numpy as np
import pandas as pd
from model.ssgpr import SSGPR
import os.path
# from utils.plots import plot_predictive_1D, visualization_experiment, visualize_data
from plot_bckp import visualization_experiment, visualize_error, analyze_experiment
from src.model_fitting.gp_common import GPDataset, read_dataset
from config.configuration_parameters import ModelFitConfig as Conf
import shutil
import os
import random
import matplotlib.pyplot as plt

def stack_csv_files(directory_path):
    dfs = []  # List to store DataFrames

    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found!")
        return None

    for filename in os.listdir(directory_path):
        if "states_inputs_out_2.csv" in filename:
            file_path = os.path.join(directory_path, filename)
            dfs.append(pd.read_csv(file_path))

    result_df = pd.concat(dfs, ignore_index=True)
    original_length = len(result_df) 
    result_df = result_df.drop_duplicates(subset=['state_in', 'input_in'], keep='first')
    final_length = len(result_df)
    print(f"{(1-final_length/original_length):.2f} % of rows were dropped out of {original_length}")
    return result_df

def get_random_csv_path(directory_path, N=5, keyword="states_inputs_out_2.csv"):
    matching_files = [filename for filename in os.listdir(directory_path) if keyword in filename]
    
    if not matching_files:
        print(f"No files found with keyword '{keyword}' in '{directory_path}'.")
        return None

    random_files = random.sample(matching_files, min(N, len(matching_files)))
    return [os.path.join(directory_path, random_file) for random_file in random_files]

def compute_velocity(s):
    # Parse the string to a list of floats
    parsed_list = [float(item) for item in s.strip('[]').split(',')]
    return np.sqrt(parsed_list[6]**2 + parsed_list[7]**2 + parsed_list[8]**2)

def balance_dataset(df, save_dir):
    # Calculate the velocity magnitude for each row
    original_length = len(df)
    df['velocity_magnitude'] = df['state_in'].apply(compute_velocity)
    
    # Filter out rows with velocity magnitudes below 5
    df = df[df['velocity_magnitude'] >= 5]
    df = df[df['velocity_magnitude'] <= 20]

    # Plotting the histogram before balancing
    plt.figure(figsize=(10, 5))
    plt.hist(df['velocity_magnitude'], bins=50, color='skyblue', edgecolor='black', alpha=0.7, label="Before Balancing")

    df['bins'] = pd.cut(df['velocity_magnitude'], bins=5, labels=False)
    min_samples = df['bins'].value_counts().min()  # This makes sure all bins are equally represented
    
    # Sample that number of rows from each bin
    dfs = []
    for bin_label in df['bins'].unique():
        sample_df = df[df['bins'] == bin_label].sample(min_samples)
        dfs.append(sample_df)
    balanced_df = pd.concat(dfs, axis=0).drop(columns=['bins']).reset_index(drop=True)

    plt.hist(balanced_df['velocity_magnitude'], bins=50, color='salmon', edgecolor='black', alpha=0.5, label="After Balancing")
    plt.xlabel('Velocity Magnitude')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(save_dir)
    plt.close()

    final_length = len(balanced_df)
    print(f"{(1-final_length/original_length):.2f} % of rows were dropped out of {original_length}")
    return balanced_df

def main(x_features, u_features, z_features, reg_y_dim, dataset_name, x_cap, hist_bins, hist_thresh, nbf, train, n_restarts, maxiter):
    save_dir = os.environ["SSGPR_PATH"] + "/data/" + dataset_name
    os.makedirs(save_dir,exist_ok=True)
    filename = 'ssgpr_model_' + str(reg_y_dim)
    save_path = os.path.join(save_dir, filename)
    print(save_path)

    if (train == 1):
        # Prepare data from dataset
        dataset_path = os.environ['FLIGHTMARE_PATH'] + "/flightmpcc/saved_training/" + dataset_name + "/traj/"
        df_train = stack_csv_files(dataset_path)
        df_train = df_train.sample(frac=0.2).reset_index(drop=True) # shuffle
        # df_train = balance_dataset(df_train, os.path.join(save_dir,f'histogram_{reg_y_dim}.png')) # doesn't seem to help
        
        # # Prepare data from csv file
        # directory_path = os.environ["FLIGHTMARE_PATH"] + "/flightmpcc/saved_training/20230910_185814-TRAIN-BEM/traj"
        # filename = "eval_id_48_states_inputs_out.csv"
        # df_train = pd.read_csv(os.path.join(directory_path, filename))
        # df_train = df_train.sample(frac=1.0).reset_index(drop=True) # shuffle

        gp_dataset = GPDataset(df_train, x_features=x_features, u_features=u_features, z_features=z_features, y_dim=reg_y_dim,
                            cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)
        gp_dataset.cluster(n_clusters=1, load_clusters=False, save_dir=save_dir, visualize_data=False)

        X = gp_dataset.get_x(cluster=0)
        Y = gp_dataset.get_y(cluster=0)
        num_points = int(len(X)*0.7)
        X_train = X[:num_points,:]
        X_test = X[num_points:,:]
        Y_train = Y[:num_points,:]
        Y_test = Y[num_points:,:]

        # Create new instance
        ssgpr = SSGPR(num_basis_functions=nbf)
        ssgpr.add_data(X_train, Y_train, X_test, Y_test)
        ssgpr.optimize(save_path, restarts=n_restarts, maxiter=maxiter, verbose=True)
        ssgpr.save(f'{save_path}.pkl')
        
        # Prediction
        mu, std, _, alpha = ssgpr.predict(X_test, sample_posterior=True)
        np.savetxt(f'{save_path}_alpha.csv', alpha, delimiter=",")

        # Evaluation
        NMSE, MNLP = ssgpr.evaluate_performance(save_path,restarts=n_restarts, maxiter=maxiter)
        print("Normalised mean squared error (NMSE): %.5f" % NMSE)
        print("Mean negative log probability (MNLP): %.5f" % MNLP)
        
        # Some plotting
        visualize_error(path=save_path, X_test=X_test, Y_test=Y_test, Xs=X_test, mu=mu,
                    stddev=std, x_vis_feats=x_features,u_vis_feats=u_features,z_vis_feats=z_features)

        dataset_files = get_random_csv_path(dataset_path, N=40)
        analyze_experiment(dataset_files,x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,
                        x_vis_feats=x_features,u_vis_feats=u_features,z_vis_feats=z_features,y_vis_feats=reg_y_dim,save_file_path=save_path,ssgpr=ssgpr)
    
    elif (train == 0):
        # Load existing instance
        ssgpr = SSGPR.load(f'{save_path}.pkl')
        
        # Some plotting
        # TODO (krmaria): draw more than one sample!
        directory_path = os.environ["FLIGHTMARE_PATH"] + "/flightmpcc/saved_training/20230910_185814-TRAIN-BEM/traj"
        filename = "eval_id_48_states_inputs_out.csv"
        test_file = os.path.join(directory_path, filename)

        visualization_experiment(test_file,x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,
                        x_vis_feats=x_features,u_vis_feats=u_features,z_vis_feats=z_features,y_vis_feats=reg_y_dim,save_file_path=save_path,ssgpr=ssgpr)
    else:
        ValueError('train variable should be 0 or 1!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=int, default="1")

    parser.add_argument("--nbf", type=int, default="20",
                        help="Number of basis functions to use for the SSGP approximation")

    parser.add_argument('--x', nargs='+', type=int, default=[])

    parser.add_argument('--u', nargs='+', type=int, default=[])

    parser.add_argument('--z', nargs='+', type=int, default=[])

    parser.add_argument("--y", type=int, default=7,
                        help="Regression Y variable. Must be an integer between 0 and 12. Velocities xyz correspond to"
                             "indices 7, 8, 9.")

    parser.add_argument("--ds_name", type=str, required=True)

    args = parser.parse_args()

    # Best so far (0.23 rmse): nbf=40, maxiter=400, sample=0.2, no dataset balancing
    # Cheap (0.25 rmse): nbf=20, maxiter=200, sample=0.2, no dataset balancing

    # Stuff from Conf
    hist_bins = Conf.histogram_bins
    hist_thresh = Conf.histogram_threshold
    x_cap = Conf.velocity_cap

    main(x_features=args.x,u_features=args.u,z_features=args.z,reg_y_dim=args.y,dataset_name=args.ds_name,
         x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,nbf=args.nbf,train=args.train,n_restarts=1,maxiter=500)
