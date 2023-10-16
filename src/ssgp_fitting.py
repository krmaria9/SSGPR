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
import random
import re
from collections import defaultdict

def select_eval_ids(directory_path, num_eval_ids):
    num_bins = 6

    lap_times = []
    eval_id_dict = defaultdict(list)
    pattern = re.compile(r"eval_(\d+).*lap0_([\d.]+).*err_0")

    # List the names of YAML files in the specified directory.
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".yaml"):
            match = pattern.search(file_name)
            if match:
                eval_id, lap_time = match.groups()
                lap_time = float(lap_time)
                lap_times.append(lap_time)
                eval_id_dict[lap_time].append({"eval_id": int(eval_id), "filename": file_name})  # Store both eval_id and filename

    # Check if any matching files were found, if not, exit early
    if not lap_times:
        print("No matching files found in the specified directory.")
        return []

    # Creating bins using histogram
    _, bin_edges = np.histogram(lap_times, bins=num_bins)
    
    selected_eval_ids = []
    selected_filenames = []

    # Loop to select eval ids such that all lap times are equally represented.
    while len(selected_eval_ids) < num_eval_ids and bin_edges.size > 1:
        for i in range(bin_edges.size - 3): # skip two highest bins (third of slowest trajectories)
            lower_edge = bin_edges[i]
            upper_edge = bin_edges[i + 1]

            # Getting eval_ids within the current bin
            bin_items = [item for lt, items in eval_id_dict.items() for item in items if lower_edge <= lt < upper_edge]
            
            if bin_items and len(selected_eval_ids) < num_eval_ids:
                selected_item = random.choice(bin_items)
                selected_eval_ids.append(selected_item["eval_id"])
                selected_filenames.append(selected_item["filename"])
                bin_items.remove(selected_item)

    # Print selected filenames
    print("Selected Filenames:")
    for filename in selected_filenames:
        print(filename)

    return selected_eval_ids

def stack_csv_files(directory_path, eval_ids):
    dfs = []  # List to store DataFrames

    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found!")
        return None

    pattern = re.compile(r"eval_id_(\d+)_states_inputs_out.csv")

    for filename in os.listdir(directory_path):
        match = pattern.match(filename)
        if match and int(match.group(1)) in eval_ids:
            file_path = os.path.join(directory_path, filename)
            dfs.append(pd.read_csv(file_path))

    if not dfs:
        print("No matching files found for the selected eval ids!")
        return None
    
    result_df = pd.concat(dfs, ignore_index=True)
    original_length = len(result_df)
    result_df = result_df.drop_duplicates(subset=['state_in', 'input_in'], keep='first')
    final_length = len(result_df)
    print(f"{(1-final_length/original_length)*100:.2f}% of rows were dropped out of {original_length}")
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
    return np.sqrt(parsed_list[7]**2 + parsed_list[8]**2 + parsed_list[9]**2)

def create_steps_ahead_matrix(X, N, M):
    """
    For a given matrix X, this function creates a new matrix by concatenating rows that are N steps ahead,
    for M times.

    :param X: The input matrix
    :param N: Steps ahead
    :param M: Number of times to concatenate
    :return: New matrix
    """
    parts = [X[i:-N*M+i] for i in range(0, N*M, N)]
    return np.hstack(parts)

def main(x_features, u_features, reg_y_dim, x_cap, hist_bins, hist_thresh, nbfs, train, n_restarts, maxiter):
    save_dir = os.environ["SSGPR_PATH"] + "/data/" + "RUN_10"
    os.makedirs(save_dir,exist_ok=True)

    unit = 30
    datasets = [
        {"dataset_name": "20230917_234713-TRAIN-BEM", "track": "splits", "eval_count": 2*unit},
        # {"dataset_name": "20230918_071749-TRAIN-BEM", "track": "lem", "eval_count": 5*unit},
        # {"dataset_name": "20230918_232905-TRAIN-BEM", "track": "marv", "eval_count": unit},
    ]

    if (train == 1):
        dfs = []

        for dataset in datasets:
            dataset_name = dataset["dataset_name"]
            eval_count = dataset["eval_count"]
            dataset_path = os.path.join(os.environ['FLIGHTMARE_PATH'], "flightmpcc/saved_training", dataset_name, "pred_5/")
            dataset_config_path = os.path.join(os.environ['FLIGHTMARE_PATH'], "flightmpcc/saved_training", dataset_name, "mpc/")
            
            eval_ids = select_eval_ids(dataset_config_path, eval_count)
            df_train = stack_csv_files(dataset_path, eval_ids)
            
            dfs.append(df_train)

        # Concatenate all DataFrames in the list into a single DataFrame
        df_train = pd.concat(dfs, ignore_index=True)
        
        # # Prepare data from csv file
        # directory_path = os.environ["FLIGHTMARE_PATH"] + "/flightmpcc/saved_training/" + dataset_name_1 + "/pred_4/"
        # filename = "eval_id_39_states_inputs_out.csv"
        # df_train = pd.read_csv(os.path.join(directory_path, filename))
        
        max_samples = 2e5
        ratio = min(1,max_samples/df_train.shape[0])
        print(ratio)
        ratio = 1
        
        df_train = df_train.sample(frac=ratio).reset_index(drop=True) # shuffle

        gp_dataset = GPDataset(df_train, x_features=x_features, u_features=u_features, y_dim=reg_y_dim,
                            cap=x_cap, n_bins=hist_bins, thresh=hist_thresh, visualize_data=False)
        gp_dataset.cluster(n_clusters=1, load_clusters=False, save_dir=save_dir, visualize_data=False)

        X_init = gp_dataset.get_x(cluster=0)
        
        for y,nbf in zip(reg_y_dim,nbfs):
            filename = 'ssgpr_model_' + str(y)
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_path,exist_ok=True)

            Y_init = gp_dataset.get_y(cluster=0, y_dim=y)

            num_points = int(min(5e4,X_init.shape[0]))
                    
            X = X_init[:num_points]
            Y = Y_init[:num_points]
            
            cut = int(num_points * 0.6) # 60 percent for training
        
            X_train = X[:cut,:]
            X_test = X[cut:,:]
            Y_train = Y[:cut,:]
            Y_test = Y[cut:,:]

            # Create new instance
            ssgpr = SSGPR(num_basis_functions=nbf)
            ssgpr.add_data(X_train, Y_train, X_test, Y_test)
            ssgpr.optimize(save_path, restarts=n_restarts, maxiter=maxiter, verbose=True)
            ssgpr.save(f'{save_path}.pkl')

            # ssgpr = SSGPR.load(f'{save_path}.pkl')
            
            # Prediction
            mu, alpha_train = ssgpr.predict_maria(Xs=X_test)
            np.savetxt(f'{save_path}_alpha_train.csv', alpha_train, delimiter=",")
            _, alpha_all = ssgpr.predict_maria(Xs=X_init, Ys=Y_init)
            np.savetxt(f'{save_path}_alpha_all.csv', alpha_all, delimiter=",")

            # # Evaluation
            # NMSE, MNLP = ssgpr.evaluate_performance(save_path,restarts=n_restarts, maxiter=maxiter)
            # print("Normalised mean squared error (NMSE): %.5f" % NMSE)
            # print("Mean negative log probability (MNLP): %.5f" % MNLP)

            # # Some plotting
            # visualize_error(path=save_path, X_test=X_test, Y_test=Y_test, Xs=X_test, mu=mu,
            #             stddev=mu, x_vis_feats=x_features,u_vis_feats=u_features,z_vis_feats=z_features)

            # dataset_files = get_random_csv_path(dataset_path, N=40)
            # analyze_experiment(dataset_files,x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,
            #                 x_vis_feats=x_features,u_vis_feats=u_features,z_vis_feats=z_features,y_vis_feats=reg_y_dim,save_file_path=save_path,ssgpr=ssgpr)
        
            # test_file = os.path.join(directory_path, filename)

            # visualization_experiment(test_file,x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,
            #                 x_vis_feats=x_features,u_vis_feats=u_features,z_vis_feats=z_features,
            #                 y_vis_feats=reg_y_dim,save_file_path=save_path,ssgpr=ssgpr)
            
            # Root path for the directories.
            root_path = os.environ["FLIGHTMARE_PATH"] + "/flightmpcc/saved_training/"
            eval_ids = [18, 21, 167, 190, 261]
            directory_path = root_path + dataset_name + "/pred_5/"
            track = dataset["track"]
            
            for dataset in datasets:
                # dataset_name = dataset["dataset_name"]
                # track = dataset["track"]
                # directory_path = root_path + dataset_name + "/pred_5/"
                # dataset_config_path = root_path + dataset_name + "/mpc/"
                # eval_ids = select_eval_ids(dataset_config_path, 5)
                # eval_ids.append(49)
                
                for eval_id in eval_ids:
                    filename = f"eval_id_{eval_id}_states_inputs_out.csv"
                    test_file = os.path.join(directory_path, filename)
                    save_file_path = save_path + f"/{track}_{eval_id}"

                    visualization_experiment(
                        test_file,
                        x_cap=x_cap,
                        hist_bins=hist_bins,
                        hist_thresh=hist_thresh,
                        x_vis_feats=x_features,
                        u_vis_feats=u_features,
                        y_vis_feats=y,
                        save_file_path=save_file_path,
                        ssgpr=ssgpr,
                        alpha_train=alpha_train,
                        alpha_all=alpha_all
                    )
        
    elif (train == 0):
        # Root path for the directories.
        root_path = os.environ["FLIGHTMARE_PATH"] + "/flightmpcc/saved_training/"
        # for dataset in datasets:
        #     dataset_name = dataset["dataset_name"]
        #     track = dataset["track"]
        #     directory_path = root_path + dataset_name + "/pred_5/"
        #     dataset_config_path = root_path + dataset_name + "/mpc/"
        #     eval_ids = select_eval_ids(dataset_config_path, 5)
        #     eval_ids.append(49)
        eval_ids = [18, 21, 167, 190, 261]

        for y in reg_y_dim:
            filename = 'ssgpr_model_' + str(y)
            save_path = os.path.join(save_dir, filename)
            os.makedirs(save_path,exist_ok=True)

            # Load existing instance
            ssgpr = SSGPR.load(f'{save_path}.pkl')
            alpha_train = np.loadtxt(f'{save_path}_alpha_train.csv')
            alpha_all = np.loadtxt(f'{save_path}_alpha_all.csv')
            os.makedirs(save_path,exist_ok=True)
            
            for dataset in datasets:
                dataset_name = dataset["dataset_name"]
                track = dataset["track"]
                directory_path = root_path + dataset_name + "/pred_5/"
                # dataset_config_path = root_path + dataset_name + "/mpc/"
                # eval_ids = select_eval_ids(dataset_config_path, 5)
                # eval_ids.append(49)
                
                for eval_id in eval_ids:
                    filename = f"eval_id_{eval_id}_states_inputs_out.csv"
                    test_file = os.path.join(directory_path, filename)
                    save_file_path = save_path + f"/{track}_{eval_id}"

                    visualization_experiment(
                        test_file,
                        x_cap=x_cap,
                        hist_bins=hist_bins,
                        hist_thresh=hist_thresh,
                        x_vis_feats=x_features,
                        u_vis_feats=u_features,
                        y_vis_feats=y,
                        save_file_path=save_file_path,
                        ssgpr=ssgpr,
                        alpha_train=alpha_train,
                        alpha_all=alpha_all
                    )
        
    else:
        ValueError('train variable should be 0 or 1!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--train', type=int, default="1")

    parser.add_argument("--nbf", nargs='+', type=int, default=[])

    parser.add_argument('--x', nargs='+', type=int, default=[])

    parser.add_argument('--u', nargs='+', type=int, default=[])

    parser.add_argument("--y", nargs='+', type=int, default=[])

    args = parser.parse_args()

    # Best so far (0.23 rmse): nbf=40, maxiter=400, sample=0.2, no dataset balancing
    # Cheap (0.25 rmse): nbf=20, maxiter=200, sample=0.2, no dataset balancing

    # Stuff from Conf
    hist_bins = Conf.histogram_bins
    hist_thresh = Conf.histogram_threshold
    x_cap = Conf.velocity_cap

    main(x_features=args.x,u_features=args.u,reg_y_dim=args.y,
         x_cap=x_cap,hist_bins=hist_bins,hist_thresh=hist_thresh,nbfs=args.nbf,train=args.train,n_restarts=1,maxiter=250)
