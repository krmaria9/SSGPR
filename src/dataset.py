"""
Adapted from https://github.com/uzh-rpg/data_driven_mpc/blob/main/ros_gp_mpc/src/model_fitting/gp_common.py
"""

import numpy as np
import json


class Dataset:
    def __init__(self, raw_ds=None,
                 x_features=None, u_features=None, y_dim=None,
                 cap=None, n_bins=None, thresh=None,
                 visualize_data=False):

        self.data_labels = [r'$p_x$', r'$p_y$', r'$p_z$', r'$q_w$', r'$q_x$', r'$q_y$', r'$q_z$',
                            r'$v_x$', r'$v_y$', r'$v_z$', r'$w_x$', r'$w_y$', r'$w_z$']

        # Raw dataset data
        self.x_raw = None
        self.x_out_raw = None
        self.u_raw = None
        self.y_raw = None
        self.x_pred_raw = None
        self.dt_raw = None

        self.x_features = x_features
        self.u_features = u_features
        self.y_dim = y_dim

        # Data pruning parameters
        self.cap = cap
        self.n_bins = n_bins
        self.thresh = thresh
        self.plot = visualize_data

        # GMM clustering
        self.pruned_idx = []
        self.cluster_agency = None
        self.centroids = None

        # Number of data clusters
        self.n_clusters = 1

        if raw_ds is not None:
            self.load_data(raw_ds)

    # TODO (krmaria): this is the time consuming operation, try to switch to casadi?? Or could to the world to body mapping in the predict part?
    def load_data(self, ds):

        x_raw = np.array([json.loads(elem) for elem in ds['state_in']])
        x_out = np.array([json.loads(elem) for elem in ds['state_out']])
        x_pred = np.array([json.loads(elem) for elem in ds['state_nom']])
        u_raw = np.array([json.loads(elem) for elem in ds['input_in']])

        dt = ds["dt"].to_numpy()
        invalid = np.where(dt == 0)

        # Remove invalid entries (dt = 0)
        x_raw = np.delete(x_raw, invalid, axis=0)
        x_out = np.delete(x_out, invalid, axis=0)
        x_pred = np.delete(x_pred, invalid, axis=0)
        u_raw = np.delete(u_raw, invalid, axis=0)
        dt = np.delete(dt, invalid, axis=0)

        # Rotate velocities to body frame and recompute errors
        # It's not the same as first substracting x_out - x_pred, and then doing the frame trafo (only if q_pred==q_out, which is not the case)
        # Only if q is the same in both cases (out, pred)
        x_raw = world_to_body_velocity_mapping(x_raw, x_raw)
        x_pred = world_to_body_velocity_mapping(x_pred, x_raw)
        x_out = world_to_body_velocity_mapping(x_out, x_raw)
        
        # Augment with zeros in case of column mismatch (i.e if N/A, let's use 0)
        additional_cols = np.zeros((x_pred.shape[0], x_out.shape[1] - x_pred.shape[1]))
        x_pred = np.hstack((x_pred, additional_cols))

        y_err = x_out - x_pred

        # Normalize error by window time (i.e. predict error dynamics instead of error itself)
        y_err /= np.expand_dims(dt, 1)

        # Select features
        self.x_raw = x_raw
        self.x_out_raw = x_out
        self.u_raw = u_raw
        self.y_raw = y_err
        self.x_pred_raw = x_pred
        self.dt_raw = dt

    def get_x(self, cluster=None, pruned=True, raw=False):

        if cluster is not None:
            assert pruned

        if raw:
            return self.x_raw[tuple(self.pruned_idx)] if pruned else self.x_raw

        data_list = []
        if self.x_features:
            x_f = self.x_features
            data_list.append(self.x_raw[:, x_f])
        if self.u_features:
            u_f = self.u_features
            data_list.append(self.u_raw[:, u_f])
        data = np.concatenate(data_list, axis=1)

        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def x(self):
        return self.get_x()

    def get_x_out(self, cluster=None, pruned=True):

        if cluster is not None:
            assert pruned

        if pruned or cluster is not None:
            data = self.x_out_raw[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

            return data

        return self.x_out_raw[tuple(self.pruned_idx)] if pruned else self.x_out_raw

    @property
    def x_out(self):
        return self.get_x_out()

    def get_u(self, cluster=None, pruned=True, raw=False):

        if cluster is not None:
            assert pruned

        if raw:
            return self.u_raw[tuple(self.pruned_idx)] if pruned else self.u_raw

        data = self.u_raw[:, self.u_features] if self.u_features is not None else self.u_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def u(self):
        return self.get_u()

    def get_y(self, y_dim=None, cluster=None, pruned=True, raw=False):
        if y_dim is not None:
            self.y_dim = y_dim

        if cluster is not None:
            assert pruned

        if raw:
            return self.y_raw[self.pruned_idx] if pruned else self.y_raw

        data = self.y_raw[:, self.y_dim] if self.y_dim is not None else self.y_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned or cluster is not None:
            data = data[tuple(self.pruned_idx)]
            data = data[self.cluster_agency[cluster]] if cluster is not None else data

        return data

    @property
    def y(self):
        return self.get_y()

    def get_x_pred(self, pruned=True, raw=False):

        if raw:
            return self.x_pred_raw[tuple(self.pruned_idx)] if pruned else self.x_pred_raw

        data = self.x_pred_raw[:, self.y_dim] if self.y_dim is not None else self.x_pred_raw
        data = data[:, np.newaxis] if len(data.shape) == 1 else data

        if pruned:
            data = data[tuple(self.pruned_idx)]

        return data

    @property
    def x_pred(self):
        return self.get_x_pred()

    def get_dt(self, pruned=True):

        return self.dt_raw[tuple(self.pruned_idx)] if pruned else self.dt_raw

    @property
    def dt(self):
        return self.get_dt()


def world_to_body_velocity_mapping(state_sequence, state_sequence_2):
    p, q, v_w, w = separate_variables(state_sequence)
    _, q_2, _, _ = separate_variables(state_sequence_2)

    q_2_inv = quaternion_inverse(q_2)

    v_b = v_dot_q(v_w, q_2_inv)
    
    return np.concatenate((p, q, v_b, w), 1)

def separate_variables(traj):
    """
    Reshapes a trajectory into expected format.

    :param traj: N x 13 array representing the reference trajectory
    :return: A list with the components: Nx3 position trajectory array, Nx4 quaternion trajectory array, Nx3 velocity
    trajectory array, Nx3 body rate trajectory array
    """

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]
    return [p_traj, a_traj, v_traj, r_traj]


def v_dot_q(v, q):
    rot_mats = q_to_rot_mat(q)
    return np.einsum('ijk,ik->ij', rot_mats, v)


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    rot_mat = np.zeros((q.shape[0], 3, 3))

    rot_mat[:, 0, 0] = 1 - 2 * (qy ** 2 + qz ** 2)
    rot_mat[:, 0, 1] = 2 * (qx * qy - qw * qz)
    rot_mat[:, 0, 2] = 2 * (qx * qz + qw * qy)

    rot_mat[:, 1, 0] = 2 * (qx * qy + qw * qz)
    rot_mat[:, 1, 1] = 1 - 2 * (qx ** 2 + qz ** 2)
    rot_mat[:, 1, 2] = 2 * (qy * qz - qw * qx)

    rot_mat[:, 2, 0] = 2 * (qx * qz - qw * qy)
    rot_mat[:, 2, 1] = 2 * (qy * qz + qw * qx)
    rot_mat[:, 2, 2] = 1 - 2 * (qx ** 2 + qy ** 2)

    return rot_mat


def quaternion_inverse(q):
    q_inv = q.copy()
    q_inv[:, 1:] *= -1
    return q_inv
