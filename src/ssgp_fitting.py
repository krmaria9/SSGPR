import argparse
import sys
import numpy as np
import pandas as pd
from model.ssgpr import SSGPR
import os.path
from plot_bckp import visualize_error, visualization_experiment
from src.model_fitting.gp_common import GPDataset
from config.configuration_parameters import ModelFitConfig as Conf
import os

# TODO (krmaria): discard episodes which failed
def stack_csv_files(directory_path):
    dfs = []  # List to store DataFrames

    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found!")
        return None

    for filename in os.listdir(directory_path):
        if "states_inputs.csv" in filename:
            file_path = os.path.join(directory_path, filename)
            dfs.append(pd.read_csv(file_path))

    result_df = pd.concat(dfs, ignore_index=True)
    original_length = len(result_df) 
    result_df = result_df.drop_duplicates(subset=['state_in', 'input_in'], keep='first')
    final_length = len(result_df)
    print(f"{(1-final_length/original_length):.2f} % of rows were dropped out of {original_length}")
    return result_df

def main(x_features, u_features, reg_y_dim, x_cap, hist_bins, hist_thresh, nbf, train, n_restarts, maxiter):
    save_dir = os.environ['FLIGHTMARE_PATH'] + "/externals/SSGPR/data/RUN_1"
    os.makedirs(save_dir,exist_ok=True)
    base_path = os.environ['FLIGHTMARE_PATH'] + "/flightmpcc/saved_training/"

    if (train == 1):
        # List of dataset names
        dataset_names = ["20230917_234713-TRAIN-BEM", "20230918_071749-TRAIN-BEM"]

        # Load and concatenate DataFrames
        df_list = []
        for dataset in dataset_names:
            dataset_path = base_path + dataset + "/nom/"
            df = stack_csv_files(dataset_path)
            df_list.append(df)

        # Concatenate all DataFrames
        df_train = pd.concat(df_list, ignore_index=True)

        max_samples = 3e5
        ratio = max_samples/df_train.shape[0]
        print(f'Take {ratio:.2f} samples from training set')
        
        df_train = df_train.sample(frac=ratio).reset_index(drop=True) # shuffle

        gp_dataset = GPDataset(df_train, x_features=x_features, u_features=u_features, y_dim=reg_y_dim,
                            cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)
        gp_dataset.cluster(n_clusters=1, load_clusters=False, save_dir=save_dir, visualize_data=False)
    
        X_init = gp_dataset.get_x(cluster=0)
        
        for y in reg_y_dim:

            filename = f'ssgpr_model_{y}'
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_path,exist_ok=True)

            Y_init = gp_dataset.get_y(cluster=0, y_dim=y)
            
            # Remove outliers
            threshold = 2  # number of std's
            mean_y = np.mean(Y_init)
            std_y = np.std(Y_init)

            mask = (Y_init > mean_y - threshold * std_y) & (Y_init < mean_y + threshold * std_y)
            mask = mask.flatten()
            
            X = X_init[mask]
            Y = Y_init[mask]

            # Split in train and validation sets
            num_points = int(len(X)*0.7)
            X_train = X[:num_points,:]
            X_val = X[num_points:,:]
            Y_train = Y[:num_points,:]
            Y_val = Y[num_points:,:]

            # Create new instance
            ssgpr = SSGPR(num_basis_functions=nbf)
            ssgpr.add_data(X_train, Y_train, X_val, Y_val)
            ssgpr.optimize(save_path, restarts=n_restarts, maxiter=maxiter, verbose=True)
            ssgpr.save(f'{save_path}.pkl')

            mu, alpha_train = ssgpr.predict_maria(Xs=X_val)
            # np.savetxt(f'{save_path}_alpha_train.csv', alpha_train, delimiter=",")
            _, alpha_all = ssgpr.predict_maria(Xs=X_init, Ys=Y_init)
            np.savetxt(f'{save_path}_alpha_all.csv', alpha_all, delimiter=",")
            
            # Some plotting
            visualize_error(path=save_path, X_test=X_val, Y_test=Y_val, Xs=X_val, mu=mu,
                        x_vis_feats=x_features,u_vis_feats=u_features)
        
            # Root path for the directories.
            eval_ids = [12, 47, 78, 120, 165]
            
            for dataset in dataset_names:
                directory_path = base_path + dataset + "/nom/"
                for eval_id in eval_ids:
                    filename = f"eval_id_{eval_id}_states_inputs.csv"
                    test_file = os.path.join(directory_path, filename)

                    visualization_experiment(
                        test_file,
                        x_cap=x_cap,
                        hist_bins=hist_bins,
                        hist_thresh=hist_thresh,
                        x_vis_feats=x_features,
                        u_vis_feats=u_features,
                        y_vis_feats=y,
                        save_file_path=f"{save_path}/{dataset}/eval_id_{eval_id}",
                        ssgpr=ssgpr,
                        alpha_all=alpha_all
                    )
        
    elif (train == 0):

        eval_ids = [12]

        for y in reg_y_dim:

            filename = f'ssgpr_model_{y}'
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_path,exist_ok=True)
            
            # Load existing instance
            ssgpr = SSGPR.load(f'{save_path}.pkl')
            alpha_all = np.loadtxt(f'{save_path}_alpha_all.csv')
            os.makedirs(save_path,exist_ok=True)
            
            for dataset in dataset_names:
                directory_path = base_path + dataset + "/nom/"
                for eval_id in eval_ids:
                    filename = f"eval_id_{eval_id}_states_inputs.csv"
                    test_file = os.path.join(directory_path, filename)

                    visualization_experiment(
                        test_file,
                        x_cap=x_cap,
                        hist_bins=hist_bins,
                        hist_thresh=hist_thresh,
                        x_vis_feats=x_features,
                        u_vis_feats=u_features,
                        y_vis_feats=y,
                        save_file_path=f"{save_path}/{dataset}/eval_id_{eval_id}",
                        ssgpr=ssgpr,
                        alpha_all=alpha_all
                    )
    else:
        ValueError('Train should be 0 or 1!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=int, default="1")

    parser.add_argument("--nbf", type=int, default="20",
                        help="Number of basis functions to use for the SSGP approximation")

    parser.add_argument('--x', nargs='+', type=int, default=[])

    parser.add_argument('--u', nargs='+', type=int, default=[])

    parser.add_argument("--y", nargs='+', type=int, default=[])

    args = parser.parse_args()

    # Stuff from Conf
    hist_bins = Conf.histogram_bins
    hist_thresh = Conf.histogram_threshold
    x_cap = Conf.velocity_cap

    main(x_features=args.x,u_features=args.u,reg_y_dim=args.y,
         x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,nbf=args.nbf,train=args.train,n_restarts=1,maxiter=250)
