#!/bin/bash

# Get the directory of the script
script_dir="$(dirname "$(realpath "$0")")"

# Get the grandparent directory of the script
grandparent_dir="$(dirname "$script_dir")"

# Add grandparent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${grandparent_dir}"

# Env variable for SSGPR
export SSGPR_PATH=${grandparent_dir}

# n_basis_functions="10 10 10 40 40 40"
# dim_indices_2="13 14 15 16 17 18" # force, torque

# n_basis_functions="40 40 40"
# dim_indices_2="13 14 15" # force, torque

# n_basis_functions="40"
# dim_indices_2="16" # taux

# n_basis_functions="40 40 40 40"
# dim_indices_2="15 16 17 18" # force, torque

n_basis_functions="40 40 40 40 40 40"
dim_indices_2="13 14 15 16 17 18" # force, torque

input_feats="4 5 6 7" # th, mot
state_feats="7 8 9 10 11 12" # v, w

python3 "$script_dir"/ssgp_fitting.py --train 0 --nbf $n_basis_functions --x $state_feats --u $input_feats --y $dim_indices_2
