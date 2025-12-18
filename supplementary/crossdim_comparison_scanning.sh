#!/usr/bin/env bash 

# If running with sbatch - remember to active the appropriate environment first, e.g. with "conda run -n <env_name>"

SCRIPT="codebase.lateral.comparison"
CMD=(python -m "$SCRIPT")

for version in low typical high; do 
    for case in A B C D E; do 
        echo "Running: ${CMD[*]} $version $case"
        "${CMD[@]}" "$version" "$case"
    done
done

echo "Finished all 3d comparisons"