#!/bin/bash

# Get the directory of the script
script_dir="$(dirname "$(realpath "$0")")"

# Get the grandparent directory of the script
grandparent_dir="$(dirname "$script_dir")"

# Add grandparent directory to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${grandparent_dir}"

# Env variable for SSGPR
export SSGPR_PATH=${grandparent_dir}

# The first argument is the number of points, default value is 100
n_basis_functions=${1:-20}

# The rest of the arguments are the dimension indices, default values are "7 8 9"
dim_indices="7 8 9" # v
state_feats="7 8 9"
for i in $dim_indices
do
    echo "Running with index $i..."
    python3 "$script_dir"/ssgp_fitting.py --train 0 --nbf $n_basis_functions --x $state_feats --y $i --ds_name "20230910_185814-TRAIN-BEM"
done

dim_indices_2="10 11 12" # w
input_feats="4 5 6 7" # mot
state_feats="7 8 9 10 11 12" # v, w
# aux_feats="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

for i in $dim_indices_2
do
    echo "Running with index $i..."
    python3 "$script_dir"/ssgp_fitting.py --train 0 --nbf $n_basis_functions --x $state_feats --u $input_feats --y $i --ds_name "20230910_185814-TRAIN-BEM"
    # python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --z $aux_feats --y $i --ds_name "20230910_185814-TRAIN-BEM"
done
