#!/usr/bin/env bash 

NP=16
SCRIPT="codebase.lateral.mpiscan"

CMD=(mpirun -n "$NP" python -m "$SCRIPT")

for rhomax in 0.3 0.2; do
    echo "Running: ${CMD[*]} --rhomax $rhomax"
    "${CMD[@]}" --rhomax "$rhomax"
done

echo "Finished all lateral scanning runs."