#!/usr/bin/env bash 

# If running with sbatch - remember to active the appropriate environment first, e.g. with "conda run -n <env_name>"

NP=16
SCRIPT="mpiscanner.py"

CMD=(mpirun -n "$NP" python "$SCRIPT")

for case in A B C D E; do
    for quantity in Ca gs K; do 
        echo "Running: ${CMD[*]} --case $case --quantity $quantity --save-series"
        "${CMD[@]}" --case "$case" --quantity "$quantity" --save-series
    done
done

echo "Finished all showcase hydration runs."