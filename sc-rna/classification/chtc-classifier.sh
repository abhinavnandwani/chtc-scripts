#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Unpacking data..."
tar -xvf scFoundation-embeddings-PASCODE.tar
rm scFoundation-embeddings-PASCODE.tar

echo "Running training script..."
bash train_cv.sh

echo "Archiving results..."
tar -cvf results_scFoundation.tar results_scFoundation

echo "Job finished at $(date)"
