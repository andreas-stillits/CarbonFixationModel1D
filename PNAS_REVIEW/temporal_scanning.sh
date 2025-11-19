#!/usr/bin/env bash 

# If running with sbatch - remember to active the appropriate environment first, e.g. with "conda run -n <env_name>"

NP=16
SCRIPT="review.temporal.mpiscanner"

CMD=(mpirun -n "$NP" python -m "$SCRIPT")

for case in A B C D E; do
    for quantity in Ca gs K; do 
        echo "Running: ${CMD[*]} --case $case --quantity $quantity --save-series"
        "${CMD[@]}" --case "$case" --quantity "$quantity" --save-series
    done
done

echo "Finished all showcase hydration runs."