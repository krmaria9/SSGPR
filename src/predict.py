import casadi as ca
import csv
import numpy as np
import os
import pandas as pd
from dynamics import dynamics, dynamics_ssgp, dynamics_simple, load_params
from plots import plot_states

STATE_DIM = 13
INPUT_DIM = 8

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    ])

def rotate_quat(q, v):
    q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    return quat_mult(quat_mult(q, np.insert(v, 0, 0)), q_conj)[1:]

def augment_x_seq(t_seq, x_seq, u_seq, p):
    augmented_x_seq = []

    for i in range(1, len(x_seq)):
        dt = t_seq[i] - t_seq[i-1]
        x_prev = x_seq[i-1]
        x = x_seq[i]

        q = x[3:7] # from body to world frame
        v = x[7:10] # linear velocity in the world frame
        w = x[10:13]

        v_prev = x_prev[7:10]
        w_prev = x_prev[10:13]

        # Compute v_dot and w_dot
        v_dot = (v - v_prev) / dt
        w_dot = (w - w_prev) / dt

        inertia = np.diag(np.array(p.inertia_diag.full()).flatten())

        # Aerodynamics
        cd = np.array(p.cd.full()).flatten()
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]]) # from world to body
        v_B = rotate_quat(q_conj, v)
        Fd_B = np.array([cd[0] * v_B[0] + cd[3] * v_B[0]**3,
                         cd[1] * v_B[1] + cd[4] * v_B[1]**3,
                         cd[2] * v_B[2] + cd[5] * v_B[2]**3 - cd[6] * (v_B[0]**2 + v_B[1]**2)])

        # Compute thrust force (body frame) and torque
        g = np.array(p.g.full()).flatten()
        v_dot_B = rotate_quat(q_conj, v_dot - g)

        Ft_B = v_dot_B * p.mass + Fd_B 
        tau = inertia @ w_dot + np.cross(w, inertia @ w)

        # Augment x with thrust force (body frame) and torque
        augmented_x = np.concatenate((x, Ft_B, tau))
        augmented_x_seq.append(augmented_x)

    return np.array(augmented_x_seq)

def load_sequence(df, dt):
    first_case_columns = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "th0", "th1", "th2", "th3", "motw", "motx", "moty", "motz"]
    # first_case_columns = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "motw", "motx", "moty", "motz"]
    # first_case_columns = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "th0", "th1", "th2", "th3"]
    is_first_case = all(col in df.columns for col in first_case_columns)

    if not is_first_case:
        raise ValueError("Wrong columns in dataframe")

    if is_first_case:
        uniform_time = np.arange(min(df['t']), max(df['t']), dt)
        df_interpolated = pd.DataFrame({
            't': uniform_time
        })
        for col in first_case_columns:
            df_interpolated[col] = np.interp(uniform_time, df['t'], df[col])

        df = df_interpolated

        t_seq = np.array(df['t'])
        x_seq = df[["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz"]].values
        u_seq = np.concatenate((df[["th0", "th1", "th2", "th3"]].values, df[["motw", "motx", "moty", "motz"]].values), axis=1)
        # u_seq = df[["motw", "motx", "moty", "motz"]].values
        # u_seq = df[["th0", "th1", "th2", "th3"]].values

    return x_seq, u_seq, t_seq

def integrate(x_seq, u_seq, integrator, fun, sample_dt=0.01, t_open_loop=600, use_ssgp=False):
    num_steps, _ = x_seq.shape
        
    x_seq_out = np.zeros((num_steps + 1, STATE_DIM))
    x_dot_seq_out = np.zeros((num_steps + 1, STATE_DIM))
    x_next = np.zeros(STATE_DIM)
    reg_dot_out = np.zeros((num_steps + 1, STATE_DIM + 6))

    t = 0 # timestamp
    for i, (x_in, u_in) in enumerate(zip(x_seq, u_seq)):
        x = x_in.copy()
        u = u_in.copy()
        
        # TODO (krmaria): cheat, pass the right residual (to make sure dynamics are correct)
        reg_in = np.zeros_like(x)
        reg_in[13:] = x[13:]

        if t > t_open_loop:
            # Initialize with prediction data
            x_init = x_seq_out[i-1,:STATE_DIM]
        else:
            # Initialize with sequence data
            x_init = x[:STATE_DIM]

        if use_ssgp:
            # Correction
            res = integrator(x0=x_init, p=np.concatenate((u, reg_in)))
            f, reg = fun(x_init, u, reg_in)

            f = np.array(f).reshape(-1) # f_nominal + reg
            reg = np.array(reg).reshape(-1) # reg
            x_next = np.array(res['xf']).reshape(-1)
        else:
            # Nominal
            res = integrator(x0=x_init, p=u_seq[i])
            f = fun(x_init, u)
            
            f = np.array(f).reshape(-1)
            x_next = np.array(res['xf']).reshape(-1)
        
        x_seq_out[i] = x_next
        x_dot_seq_out[i] = f
        t += sample_dt

        if use_ssgp:
            reg_dot_out[i] = reg
        
        if np.isnan(f).any():
            raise ValueError(f"f invalid at iteration {i}, t = {t}")

        if np.isnan(x_next).any():
            raise ValueError(f"x_next invalid at iteration {i}, t = {t}")

    return x_seq_out, x_dot_seq_out, reg_dot_out

def format_to_csv(x_seq, u_seq, x_out_nominal, x_dot_out_nominal, reg_dot_out_nominal, x_out_ssgp, x_dot_out_ssgp, reg_dot_out_ssgp, x_out_simple, x_dot_out_simple, reg_dot_out_simple, t_seq, output_file, model_flags):

    def convert_value(val):
        if isinstance(val, (ca.DM, ca.MX)):
            val = val.full().flatten().tolist()
        formatted_values = [f"{x:.{5}g}" for x in val]
        return "[" + ", ".join(formatted_values) + "]"

    fieldnames = ['index', 'timestamp', 'dt', 'state_in', 'input_in', 'state_out', 'state_nom', 'state_dot_nom', 'reg_dot_nom']

    if model_flags['USE_SSGP']:
        fieldnames.append('state_ssgp')
        fieldnames.append('state_dot_ssgp')
        fieldnames.append('reg_dot_ssgp')

    if model_flags['USE_SIMPLE']:
        fieldnames.append('state_simple')
        fieldnames.append('state_dot_simple')
        fieldnames.append('reg_dot_simple')
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(t_seq.shape[0]-2):
            row_data = {
                'index': i,
                'timestamp': f"{t_seq[i]:.{5}g}",
                'dt': f"{(t_seq[i+1] - t_seq[i]):.{5}g}",
                'state_in': convert_value(x_seq[i]),
                'input_in': convert_value(u_seq[i]),
                'state_out': convert_value(x_seq[i+1]),
                'state_nom': convert_value(x_out_nominal[i]),
                'state_dot_nom': convert_value(x_dot_out_nominal[i]),
                'reg_dot_nom': convert_value(reg_dot_out_nominal[i])
            }

            if model_flags['USE_SSGP']:
                row_data['state_ssgp'] = convert_value(x_out_ssgp[i]) # integral(f_nominal+reg)
                row_data['state_dot_ssgp'] = convert_value(x_dot_out_ssgp[i]) # f_nominal+reg
                row_data['reg_dot_ssgp'] = convert_value(reg_dot_out_ssgp[i]) # reg

            if model_flags['USE_SIMPLE']:
                row_data['state_simple'] = convert_value(x_out_simple[i]) # integral(f_nominal+reg)
                row_data['state_dot_simple'] = convert_value(x_dot_out_simple[i]) # f_nominal+reg
                row_data['reg_dot_simple'] = convert_value(reg_dot_out_simple[i]) # reg
            
            writer.writerow(row_data)

def predict_dynamics(input_file, output_file, model_flags, sample_dt, t_open_loop):

    if not os.path.exists(input_file):
        print(f"Skipping {input_file} since file does not exist.")
        return

    df = pd.read_csv(input_file, index_col=0)
    x_seq, u_seq, t_seq = load_sequence(df, sample_dt)

    params_file = os.environ["AGI_PATH"] + "/agiros/agiros/parameters/quads/kingfisher.yaml"
    params = load_params(params_file)

    # params.ssgp_path = os.environ["FLIGHTMARE_PATH"] + "/externals/SSGPR/data/RUN_4" # th
    params.ssgp_path = os.environ["FLIGHTMARE_PATH"] + "/externals/SSGPR/data/RUN_5" # mot

    # Augment state with thrust force and torque
    x_seq = augment_x_seq(t_seq, x_seq, u_seq, params)

    # Prepare integrator
    x = ca.MX.sym('x', STATE_DIM)
    reg = ca.MX.sym('reg', STATE_DIM + 6)
    u = ca.MX.sym('u', INPUT_DIM)
    h = sample_dt
    internal_dt = 1e-3 # s
    opts = {'number_of_finite_elements': int(h/internal_dt)}

    # Nominal model
    dae_nominal = {'x': x, 'p': u, 'ode': dynamics(x, u, params)}
    integrator_nominal = ca.integrator('integrator', 'rk', dae_nominal, 0, h, opts)
    f_nom = dynamics(x, u, params)
    fun_nominal = ca.Function('dynamics_fun', [x, u], [f_nom])
    x_out_nominal, x_dot_out_nominal, reg_dot_out_nominal = integrate(x_seq, u_seq, integrator_nominal, fun_nominal, sample_dt=sample_dt, t_open_loop=t_open_loop)
    x_out_ssgp = np.zeros_like(x_out_nominal)
    x_dot_out_ssgp = np.zeros_like(x_out_nominal)
    reg_dot_out_ssgp = np.zeros_like(reg_dot_out_nominal)
    x_out_simple = np.zeros_like(x_out_nominal)
    x_dot_out_simple = np.zeros_like(x_out_nominal)
    reg_dot_out_simple = np.zeros_like(reg_dot_out_nominal)

    # Use SSGP correction
    if model_flags['USE_SSGP']:
        dae_ssgp = {'x': x, 'p': ca.vertcat(u, reg), 'ode': dynamics_ssgp(x, u, params, reg)}
        integrator_ssgp = ca.integrator('integrator', 'rk', dae_ssgp, 0, h, opts)

        f_expl, reg_out = dynamics_ssgp(x, u, params, reg, ret_reg=True)
        fun_ssgp = ca.Function('dynamics_correc_fun', [x, u, reg], [f_expl, reg_out])
        x_out_ssgp, x_dot_out_ssgp, reg_dot_out_ssgp = integrate(x_seq, u_seq, integrator_ssgp, fun_ssgp, sample_dt=sample_dt, t_open_loop=t_open_loop, use_ssgp=True)

    # Simple thrust torque
    if model_flags['USE_SIMPLE']:
        dae_simple = {'x': x, 'p': u, 'ode': dynamics_simple(x, u, params)}
        integrator_simple = ca.integrator('integrator', 'rk', dae_simple, 0, h, opts)

        f_expl = dynamics_simple(x, u, params)
        fun_simple = ca.Function('dynamics_simple_fun', [x, u], [f_expl])
        x_out_simple, x_dot_out_simple, reg_dot_out_simple = integrate(x_seq, u_seq, integrator_simple, fun_simple, sample_dt=sample_dt, t_open_loop=t_open_loop)

    format_to_csv(x_seq, u_seq, x_out_nominal, x_dot_out_nominal, reg_dot_out_nominal, x_out_ssgp, x_dot_out_ssgp, reg_dot_out_ssgp, x_out_simple, x_dot_out_simple, reg_dot_out_simple, t_seq, output_file, model_flags)

    return

def main():

    model_flags = {
    'USE_SSGP': True,
    'USE_SIMPLE': True
    }

    plot_flags = {
    'ERROR': False, # out - method -> out - ssgp should be zero, out - nom is what we have to learn
    'STATE': True, # method
    'CORRECT': False, # method - nom -> ssgp - nom should be same as out - nom -> what we actually learned
    'DERIV': False, # method_dot
    'CORRECT_DERIV': False, # method_dot - nom_dot -> ssgp_dot - nom_dot should be same as out_dot - nom_dot -> should be reg_ssgp
    'XY': False
    }

    freq = 100 # Hz
    sample_dt = 1/freq # s, the interpolation timestep, i.e. timestep between two samples

    t_open_loop = 8 # s, timestamp at which the open loop prediction starts
    
    # List of dataset names
    dataset_names = ["20231122_183045-TRAIN-BEM", "20231122_194206-TRAIN-BEM"]

    # Base path
    base_path = os.environ['FLIGHTMARE_PATH'] + "/flightmpcc/saved_training/"

    # # Iterate over all files in the directory
    # for dataset_name in dataset_names:
    #     dataset_path = base_path + dataset_name + "/traj"
    #     for filename in os.listdir(dataset_path):
    #         if "states_inputs.csv" in filename:
    #             input_file = os.path.join(dataset_path, filename)
    #             if model_flags['USE_SSGP']:
    #                 output_file = input_file.replace("traj","ssgp_mot")
    #             else:
    #                 output_file = input_file.replace("traj","nom_mot")
                    
    #             # Create the directory if it doesn't exist
    #             os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
    #             predict_dynamics(input_file, output_file, model_flags, sample_dt, t_open_loop)
    #             print(output_file)

    eval_ids = [12, 47, 78, 120, 165]

    for dataset_name in dataset_names:
        dataset_path = base_path + dataset_name + "/traj"
        for eval_id in eval_ids:
            filename = f"eval_id_{eval_id}_states_inputs.csv"
            input_file = os.path.join(dataset_path, filename)
            if model_flags['USE_SSGP']:
                output_file = input_file.replace("traj","ssgp_mot")
            else:
                output_file = input_file.replace("traj","nom_mot")
                
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            predict_dynamics(input_file, output_file, model_flags, sample_dt, t_open_loop)
            plot_states(output_file, plot_flags, states_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            # plot_states(output_file, plot_flags, states_indices=[13, 14, 15, 16, 17, 18])
            print(output_file)

if __name__ == "__main__":
    main()