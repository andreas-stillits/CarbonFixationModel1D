#!/usr/bin/env bash 

# If running with sbatch - remember to active the appropriate environment first, e.g. with "conda run -n <env_name>"

NP=16
SCRIPT="codebase.temporal.mpiscan"

CMD=(mpirun -n "$NP" python -m "$SCRIPT")

for rhomax in 0.3 0.2; do
    for quantity in Ca gs K; do 
        for case in A B D E; do 
            echo "Running: ${CMD[*]} $quantity $case --rhomax $rhomax --resolution 2"
            "${CMD[@]}" "$quantity" "$case" --rhomax "$rhomax" --resolution 2
        done
        echo "Running case C with higher resolution"
        "${CMD[@]}" "$quantity" C --rhomax "$rhomax" --resolution 25
    done
done


echo "Finished all temporal scans."