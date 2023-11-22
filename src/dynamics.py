import casadi as ca
import numpy as np
import os
import yaml

loaded_ssgp_frequencies_cache = None
loaded_ssgp_coeffs_cache = None

class ParamsStruct:
    pass

def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return ca.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = ca.vertcat(
            ca.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            ca.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            ca.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return ca.vertcat(w, -x, -y, -z)

def quat_mult(q1,q2):
    ans = ca.vertcat(q2[0,:] * q1[0,:] - q2[1,:] * q1[1,:] - q2[2,:] * q1[2,:] - q2[3,:] * q1[3,:],
                  q2[0,:] * q1[1,:] + q2[1,:] * q1[0,:] - q2[2,:] * q1[3,:] + q2[3,:] * q1[2,:],
                  q2[0,:] * q1[2,:] + q2[2,:] * q1[0,:] + q2[1,:] * q1[3,:] - q2[3,:] * q1[1,:],
                  q2[0,:] * q1[3,:] - q2[1,:] * q1[2,:] + q2[2,:] * q1[1,:] + q2[3,:] * q1[0,:])
    return ans

def rotate_quat(q1,v1):
    ans = quat_mult(quat_mult(q1, ca.vertcat(0, v1)), ca.vertcat(q1[0,:],-q1[1,:], -q1[2,:], -q1[3,:]))
    return ca.vertcat(ans[1,:], ans[2,:], ans[3,:]) # to convert to 3x1 vec

def load_params(yaml_path):

    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    p = ParamsStruct()
    
    p.mass = data['mass']
    p.g = ca.vertcat(0, 0, -9.8066)
    p.inertia_diag = ca.vertcat(*data['inertia'])
    p.l_x = ca.vertcat(data['tbm_fr'][0], data['tbm_bl'][0], data['tbm_br'][0], data['tbm_fl'][0])
    p.l_y = ca.vertcat(data['tbm_fr'][1], data['tbm_bl'][1], data['tbm_br'][1], data['tbm_fl'][1])
    p.kappa = data['kappa']
    p.cd = ca.vertcat(*data['aero_coeff_1'], *data['aero_coeff_3'], data['aero_coeff_h'])
    p.motor_omega_max = data['motor_omega_max']
    p.motor_omega_min = data['motor_omega_min']
    p.reg_y = data["reg_y"]
    p.t_BM = ca.horzcat(ca.vertcat(*data['tbm_fr']),
                    ca.vertcat(*data['tbm_bl']),
                    ca.vertcat(*data['tbm_br']),
                    ca.vertcat(*data['tbm_fl']))
    
    return p

def load_ssgp_frequencies(y_dim, directory):
    global loaded_ssgp_frequencies_cache
    
    if loaded_ssgp_frequencies_cache is None:
        W_list = []
        for y in y_dim:
            file_path = os.path.join(directory, f'ssgpr_model_{y}_freqs.csv')
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found!")

            column = np.loadtxt(file_path, delimiter=",")
            W_list.append(column)

        loaded_ssgp_frequencies_cache = W_list

    return loaded_ssgp_frequencies_cache

def load_ssgp_coeffs(y_dim, directory):
    global loaded_ssgp_coeffs_cache
    
    if loaded_ssgp_coeffs_cache is None:
        mu_list = []
        for y in y_dim:
            file_path = os.path.join(directory, f'ssgpr_model_{y}_alpha_all.csv')
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{file_path}' not found!")

            column = np.loadtxt(file_path, delimiter=",")
            mu_list.append(column)

        loaded_ssgp_coeffs_cache = mu_list

    return loaded_ssgp_coeffs_cache

def phi_function(x, u, W, y):

    # Extracting the relevant parts of x_seq and u_seq
    q = x[3:7]
    v = x[7:10]
    w = x[10:13]
    v_B = v_dot_q(v, quaternion_inverse(q))

    # Constructing the output based on the value of y
    if y in [13, 14, 15, 16, 17, 18]: # force, torque
        input_feats = ca.vertcat(v_B, w, u[4:])
    else:
        raise ValueError('y not in valid reg_y range')

    NBF = W.shape[0] # number of basis functions
    phi = ca.SX.zeros(2 * NBF)
    phi[:NBF] = ca.cos(ca.mtimes(W, input_feats))
    phi[NBF:] = ca.sin(ca.mtimes(W, input_feats))

    return phi

def dynamics(x, u, p):

    q = x[3:7] # From body to world
    v = x[7:10]
    w = x[10:13]

    inertia = ca.SX.eye(3)
    inertia[0,0] = p.inertia_diag[0]
    inertia[1,1] = p.inertia_diag[1]
    inertia[2,2] = p.inertia_diag[2]

    inertia_inv = ca.SX.eye(3)
    inertia_inv[0,0] = 1/p.inertia_diag[0]
    inertia_inv[1,1] = 1/p.inertia_diag[1]
    inertia_inv[2,2] = 1/p.inertia_diag[2]

    # Aerodynamics:
    q_conj = ca.vertcat(q[0,:], -q[1,:], -q[2, :], -q[3, :]) # From world to body
    v_B = rotate_quat(q_conj, v)
    Fd_B = ca.vertcat(p.cd[0] * v_B[0] + p.cd[3] * v_B[0]**3,
                      p.cd[1] * v_B[1] + p.cd[4] * v_B[1]**3,
                      p.cd[2] * v_B[2] + p.cd[5] * v_B[2]**3 - p.cd[6] * (v_B[0]**2 + v_B[1]**2))
    Fd_W = rotate_quat(q, Fd_B)

    f_nominal = ca.vertcat(
        v,                               # p_dot
        0.5*quat_mult(q, ca.vertcat(0, w)),  # q_dot
        rotate_quat(q, ca.vertcat(0, 0, 0)) + p.g - Fd_W/p.mass,  # v_dot
        ca.mtimes(inertia_inv, ca.vertcat(                                 # w_dot
            0,
            0,
            0)
                  -ca.cross(w, ca.mtimes(inertia,w)))
    )
    
    return f_nominal

def dynamics_simple(x, u, p):

    q = x[3:7] # From body to world
    v = x[7:10]
    w = x[10:13]

    inertia = ca.SX.eye(3)
    inertia[0,0] = p.inertia_diag[0]
    inertia[1,1] = p.inertia_diag[1]
    inertia[2,2] = p.inertia_diag[2]

    inertia_inv = ca.SX.eye(3)
    inertia_inv[0,0] = 1/p.inertia_diag[0]
    inertia_inv[1,1] = 1/p.inertia_diag[1]
    inertia_inv[2,2] = 1/p.inertia_diag[2]

    # Aerodynamics:
    q_conj = ca.vertcat(q[0,:], -q[1,:], -q[2, :], -q[3, :]) # From world to body
    v_B = rotate_quat(q_conj, v)
    Fd_B = ca.vertcat(p.cd[0] * v_B[0] + p.cd[3] * v_B[0]**3,
                      p.cd[1] * v_B[1] + p.cd[4] * v_B[1]**3,
                      p.cd[2] * v_B[2] + p.cd[5] * v_B[2]**3 - p.cd[6] * (v_B[0]**2 + v_B[1]**2))
    Fd_W = rotate_quat(q, Fd_B)
    
    # Moments
    t_BM = ca.horzcat(-p.l_x, p.l_y)
    T = ca.vertcat(u[0], u[1], u[2], u[3])
    tau_yx = ca.mtimes(t_BM.T, T)

    f_nominal = ca.vertcat(
        v,                               # p_dot
        0.5*quat_mult(q, ca.vertcat(0, w)),  # q_dot
        rotate_quat(q, ca.vertcat(0, 0, (u[0]+u[1]+u[2]+u[3])/p.mass)) + p.g - Fd_W/p.mass,  # v_dot
        ca.mtimes(inertia_inv, ca.vertcat(                                 # w_dot
            tau_yx[1],
            tau_yx[0],
            p.kappa*(-u[0]-u[1]+u[2]+u[3]))
                  -ca.cross(w, ca.mtimes(inertia,w)))
    )

    return f_nominal

def dynamics_ssgp(x, u, p, reg, ret_reg=False):

    f_nominal = dynamics(x, u, p)
    
    STATE_DIM = 13

    # reg_out = ca.SX(reg)
    
    reg_out = ca.SX.zeros(reg.shape[0])

    for idx, y in enumerate(p.reg_y):

        W = load_ssgp_frequencies(p.reg_y, p.ssgp_path)[idx]
        alpha = load_ssgp_coeffs(p.reg_y, p.ssgp_path)[idx]
        phi = phi_function(x, u, W, y)

        reg_out[y] = ca.mtimes(phi.T, alpha) * 1e-2

    if all(elem in p.reg_y for elem in [13, 14, 15]): # force
        q = x[3:7]
        reg_out[7:10] += v_dot_q(reg_out[13:16], q)/p.mass # v
    
    if all(elem in p.reg_y for elem in [16, 17, 18]): # torque
        inertia_inv = ca.SX.eye(3)
        inertia_inv[0,0] = 1/p.inertia_diag[0]
        inertia_inv[1,1] = 1/p.inertia_diag[1]
        inertia_inv[2,2] = 1/p.inertia_diag[2]
        reg_out[10:13] += ca.mtimes(inertia_inv, reg_out[16:19]) # w

    f_expl = f_nominal + reg_out[:STATE_DIM]

    if ret_reg:
        return f_expl, reg_out
    else:
        return f_expl
