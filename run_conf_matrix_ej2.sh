#!/bin/bash

kernels=("linear" "sigmoid" "rbf" "poly")
c_values=("0.5" "1" "2.5" "5" "10")

for kernel in "${kernels[@]}"; do
    for c_value in "${c_values[@]}"; do
        echo "Running with kernel=$kernel and c_value=$c_value"
        python3 ej2_confusion_matrix.py "$kernel" "$c_value"
    done
done