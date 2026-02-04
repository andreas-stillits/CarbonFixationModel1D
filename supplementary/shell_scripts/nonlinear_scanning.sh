#!/usr/bin/env bash 

NP=16
SCRIPT="codebase.nonlinear.mpiscan"

CMD=(mpirun -n "$NP" python -m "$SCRIPT")

for mu in 10; do
    echo "Running: ${CMD[*]} --mu $mu"
    "${CMD[@]}" --mu "$mu"
done

echo "Finished all nonlinear scanning runs."