#!/bin/bash

# Get the directory of the script
script_dir="$(dirname "$(realpath "$0")")"

# Get the grandparent directory of the script
grandparent_dir="$(dirname "$script_dir")"

# Add grandparent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${grandparent_dir}"

n_basis_functions="10"

input_feats="0 1 2 3" # th
state_feats="7 8 9 10 11 12" # v, w
output_feats="13 14 15 16 17 18" # force, torque

python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $state_feats --u $input_feats --y $output_feats