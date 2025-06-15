#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Unpacking data..."
tar -xf scFoundation_extracted.tar  
rm scFoundation_extracted.tar  

echo "Running metrics computation..."
python3 pseudotime.py

echo "Archiving results..."
tar -cvf results_scFoundation_pseudotime_r01x.tar results_scFoundation_pseudotime_r01x


echo "Job finished at $(date)"
