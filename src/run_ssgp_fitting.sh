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
    python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $dimension_indices --y $i --ds_name "20230903_132333-TEST-BEM"
done

# dimension_indices="3 4 5 6 7 8 9 10 11 12"
# dim_indices_2="10 11 12"
# for i in $dim_indices_2
# do
#     echo "Running with index $i..."
#     python3 "$script_dir"/ssgp_fitting.py --train 1 --nbf $n_basis_functions --x $dimension_indices --y $i --ds_name "20230903_132333-TEST-BEM"
# done
