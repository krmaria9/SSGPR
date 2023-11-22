import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

columns_state = [
    "px", "py", "pz",
    "qw", "qx", "qy", "qz",
    "vx", "vy", "vz",
    "wx", "wy", "wz",
    "fx", "fy", "fz",
    "taux", "tauy", "tauz"
]

columns_input = [
    "t",
    "th1", "th2", "th3", "th4",
    "motw", "motx", "moty", "motz"
]

def compute_rmse(predictions, ground_truth, idx):
    rmse = np.sqrt(np.mean([(pred[idx] - gt[idx])**2 for pred, gt in zip(predictions, ground_truth)]))
    return rmse

def plot_states(output_file, plot_flags, states_indices=None):
    # Ensure the dataset exists
    if not os.path.exists(output_file):
        print(f"Skipping {output_file} since file does not exist.")
        return
    
    # Read CSV
    df = pd.read_csv(output_file, index_col=0)

    # Convert the timestamp to a list
    timestamp = df['timestamp']
    
    for col in df.columns:
        # Check if the column content is a string representation of a list
        if df[col].dtype == object and df[col].str.startswith('[').any() and df[col].str.endswith(']').any():
            df[col] = df[col].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Identify the available state_columns
    methods = [col for col in df.columns if col.startswith('state_') and col != 'state_out' and col != 'state_in']

    # t_end = np.where(timestamp >= 6)[0]
    # ekf_transitions = [t_end[0]]

    # Directory to save plots
    dir_name = os.path.join(os.path.dirname(output_file), os.path.splitext(os.path.basename(output_file))[0])
    os.makedirs(dir_name, exist_ok=True)

    # Plot each state
    for idx in range(len(df['state_out'][0])):
        if states_indices and idx not in states_indices:
            continue
        
        # # Center plot around transiion line
        # if 'state_correc' in methods and ekf_transitions[0]!=0:
        #     x_min = max(0,timestamp[ekf_transitions[0]] - 10)
        #     x_max = min(timestamp.iloc[-1],timestamp[ekf_transitions[0]] + 10)
        # else:
        #     x_min = timestamp[0]
        #     x_max = timestamp.iloc[-1]
            
        x_min = timestamp[0]
        x_max = timestamp.iloc[-1]
        
        # Error = out - method -> should be close to zero when we use model learning
        if plot_flags['ERROR']:
            plt.figure()
            legends = []
            for method in methods:
                if 'dot' not in method:
                    label = method.replace('state_','')
                    plt.plot(timestamp, [(out[idx]-state[idx]) for out,state in zip(df['state_out'],df[method])])
                    legends.append(f'out - {label} (RMSE: {compute_rmse(df[method], df["state_out"], idx):.5f})')
            # plt.axvline(x=timestamp[ekf_transitions[0]], linestyle='--', color='r')
            # plt.axvline(x=timestamp[ekf_transitions[0]]+dt, linestyle='--', color='r')
            plt.xlabel('Timestamp [s]')
            plt.ylabel(columns_state[idx])
            plt.xlim([x_min,x_max])
            plt.grid(True)
            plt.legend(legends, loc='best')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/err_{columns_state[idx]}.png', dpi=400)
            plt.close()
        
        # State
        if plot_flags['STATE']:
            plt.figure()
            legends = []
            for method in methods:
                if 'dot' not in method:
                    label = method.replace('state_','')
                    plt.plot(timestamp, [state[idx] for state in df[method]])
                    legends.append(f'{label} (RMSE: {compute_rmse(df[method], df["state_out"], idx):.5f})')
                
            plt.plot(timestamp, [state[idx] for state in df['state_out']], '--')
            legends.append('out')
            # plt.axvline(x=timestamp[ekf_transitions[0]], linestyle='--', color='r')
            # plt.axvline(x=timestamp[ekf_transitions[0]]+dt, linestyle='--', color='r')
            plt.xlabel('Timestamp [s]')
            plt.ylabel(columns_state[idx])
            plt.xlim([x_min,x_max])
            plt.grid(True)
            plt.legend(legends, loc='best')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/{columns_state[idx]}.png', dpi=400)
            plt.close()

        # Correction = method - nom -> ssgp-nom, out-nom
        if plot_flags['CORRECT']:
            plt.figure()
            legends = []
            for method in methods:
                if not any(substring in method for substring in ['dot', 'nom']):
                    label = method.replace('state_','')
                    plt.plot(timestamp, [(state[idx]-nom[idx]) for state,nom in zip(df[method],df['state_nom'])])
                    legends.append(f'{label}-nom')
                
            plt.plot(timestamp, [(out[idx]-nom[idx]) for out,nom in zip(df['state_out'],df['state_nom'])], '--')
            legends.append(f'out-nom')
            # plt.axvline(x=timestamp[ekf_transitions[0]], linestyle='--', color='r')
            # plt.axvline(x=timestamp[ekf_transitions[0]]+dt, linestyle='--', color='r')
            plt.xlabel('Timestamp [s]')
            plt.ylabel(columns_state[idx])
            plt.xlim([x_min,x_max])
            plt.grid(True)
            plt.legend(legends, loc='best')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/correc_{columns_state[idx]}.png', dpi=400)
            plt.close()
        
        # Derivative terms
        if plot_flags['DERIV']:
            plt.figure()
            legends = []
            for method in methods:
                if 'dot' in method:
                    label = method.replace('state_','')
                    plt.plot(timestamp, [state[idx] for state in df[method]])
                    legends.append(label)

            plt.plot(timestamp, [(state_out[idx]-state_in[idx])/timestamp.diff().mean() for state_out, state_in in zip(df['state_out'], df['state_in'])],'--')
            legends.append('dot_out')
            # plt.axvline(x=timestamp[ekf_transitions[0]], linestyle='--', color='r')
            # plt.axvline(x=timestamp[ekf_transitions[0]]+dt, linestyle='--', color='r')
            plt.xlabel('Timestamp [s]')
            plt.ylabel(columns_state[idx])
            plt.xlim([x_min,x_max])
            plt.grid(True)
            plt.legend(legends, loc='best')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/derivative_{columns_state[idx]}.png', dpi=400)
            plt.close()

        # Correction derivative = method_dot - nom_dot
        if plot_flags['CORRECT_DERIV']:
            plt.figure()
            legends = []
            plt.plot(timestamp, [reg_dot[idx] for reg_dot in df['reg_dot_nom']], linestyle='--')
            legends.append('reg_dot_nom') # true -> ideal correction

            # plt.plot(timestamp, [(state_out[idx]-state_in[idx])/timestamp.diff().mean()-state_ssgp[idx]
            #                      for state_out, state_in, state_ssgp in zip(df['state_out'], df['state_in'], df['state_dot_ssgp'])])
            # legends.append('dot_out - dot_ssgp') # out - ssgp -> should be zero (i.e. after applying correction)

            plt.plot(timestamp, [(state_out[idx]-state_in[idx])/timestamp.diff().mean()-state_dot_nom[idx]
                                 for state_out, state_in, state_dot_nom in zip(df['state_out'], df['state_in'], df['state_dot_nom'])])
            legends.append('dot_out - dot_nom') # out - nom -> should be the same as reg_dot_ssgp
            
            # plt.axvline(x=timestamp[ekf_transitions[0]], linestyle='--', color='r')
            # plt.axvline(x=timestamp[ekf_transitions[0]]+dt, linestyle='--', color='r')
            plt.xlabel('Timestamp [s]')
            plt.ylabel(columns_state[idx])
            plt.xlim([x_min,x_max])
            plt.grid(True)
            plt.legend(legends, loc='best')
            plt.tight_layout()
            plt.savefig(f'{dir_name}/correc_derivative_{columns_state[idx]}.png', dpi=400)
            plt.close()

        # # Correction derivative = method_dot - nom_dot
        # plt.figure()
        # legends = []
        # plt.plot(timestamp, [reg_dot[idx] for reg_dot in df['reg_dot_nom']], linestyle='--')
        # legends.append('reg_dot_nom') # true -> ideal correction
        
        # plt.plot(timestamp, [reg_dot[idx] for reg_dot in df['reg_dot_ssgp']], linestyle='--')
        # legends.append('reg_dot_ssgp') # true -> ideal correction

        # plt.plot(timestamp, [state_out[idx] for state_out in df['state_out']])
        # legends.append('dot_out') # out - nom -> should be the same as reg_dot_ssgp

        # plt.xlabel('Timestamp [s]')
        # # plt.ylabel(columns_state[idx])
        # plt.xlim([x_min,x_max])
        # plt.grid(True)
        # plt.legend(legends, loc='best')
        # plt.tight_layout()
        # plt.savefig(f'{dir_name}/reg_dot_{columns_state[idx]}.png', dpi=400)
        # plt.close()

    # Plot X,Y
    if plot_flags['XY']:
        plt.figure()
        for method in methods:
            label = method.replace('state_','')
            x, y = zip(*[state[:2] for state in df[method]])
            plt.plot(x, y,label=label)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{dir_name}/xy.png', dpi=400)
        plt.close()

def main():

    # parser = argparse.ArgumentParser()

    # parser.add_argument("--bag_file", type=str, required=True)

    # args = parser.parse_args()

    bag_file = os.environ['FLIGHTMARE_PATH'] + "/misc/data_sihao/INDI_CPC15_2021-03-18-18-05-23.bag"

    dir = os.path.dirname(bag_file)
    output_file = os.path.join(dir, os.path.basename(bag_file).replace(".bag", ".csv"))

    plot_states(output_file, states_indices=[0, 1, 2, 7, 8, 9])

if __name__ == "__main__":
    main()
