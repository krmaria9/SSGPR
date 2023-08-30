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
n_basis_functions=${1:-5}

# The rest of the arguments are the dimension indices, default values are "7 8 9"
dimension_indices=${@:2}
if [ -z "$dimension_indices" ]
then
    dimension_indices="7 8 9"
fi

echo "Dimension indices: $dimension_indices"

# TODO (kmaria): shouldn't x be [7,8,9] while y is 7,8 or 9?
for i in $dimension_indices
do
    echo "Running with index $i..."
    python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $dimension_indices --y $i --ds_name "INDI_CPC15_2021-03-18-18-05-23"
done

# python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $dimension_indices --y "7" --ds_name "INDI_CPC15_2021-03-18-18-05-23"
