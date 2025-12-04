#!/usr/bin/env bash 

NP=16
SCRIPT="review.nonlinear.mpiscan"

CMD=(mpirun -n "$NP" python -m "$SCRIPT")

for mu in 0.00 0.50 1.00 2.00 4.00 8.00; do
    echo "Running: ${CMD[*]} --mu $mu"
    "${CMD[@]}" --mu "$mu"
done

echo "Finished all nonlinear scanning runs."