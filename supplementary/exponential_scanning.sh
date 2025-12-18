#!/usr/bin/env bash 

# If running with sbatch - remember to active the appropriate environment first, e.g. with "conda run -n <env_name>"

NP=16
SCRIPT="codebase.steady.mpiscan"

CMD=(mpirun -n "$NP" python -m "$SCRIPT")


echo "Running: ${CMD[*]}"
"${CMD[@]}"

echo "Finished scanning for sensitivity "