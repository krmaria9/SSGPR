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
n_basis_functions=${1:-10}

# The rest of the arguments are the dimension indices, default values are "7 8 9"
dimension_indices=${@:2}
if [ -z "$dimension_indices" ]
then
    dimension_indices="7 8 9"
fi

for i in $dimension_indices
do
    echo "Running with index $i..."
    python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $dimension_indices --y $i --ds_name "20230904_132333-TEST-BEM"
done

dim_indices_2="10 11 12" # w
input_feats="0 1 2 3" # th
state_feats="10 11 12" # w

for i in $dim_indices_2
do
    echo "Running with index $i..."
    python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $state_feats --u $input_feats --y $i --ds_name "20230904_132333-TEST-BEM"
done
