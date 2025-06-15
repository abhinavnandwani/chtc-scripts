#!/usr/bin/env bash


# Exit immediately if any command fails
set -e

echo "Unpacking data..."
tar -xf scFoundation_extracted.tar  
rm scFoundation_extracted.tar  

####### TRAINING SCRIPT STARTS #########

PYTHON_CMD="python3 train_cv.py"
DATA_PATH="scFoundation/extracted_cell_embeddings/"
FM_MODEL="scFoundation" #scMulan, UCE, Geneformer
MODEL_TYPE="mlp" #logistic
META_PATH="r03x_split_seed42.pkl"
PHENOTYPE_COLUMN="r03x"

OUT_DIR="./results_scFoundation_r03x"
SUMMARY_CSV="${OUT_DIR}/hyperparam_sweep_summary.csv"
mkdir -p "$OUT_DIR"
N_VAL_SPLITS=5

# Hyperparameter grid
BATCH_SIZES=(64 128 256) # Try lower batch sizes for donor-level classification (e.g., 16,32)
LRS=(0.0001 0.001 0.01) 
# EPOCHS=(25 50 100) # Try fewer epochs (e.g., 10)
# HIDDENS=("256,64" "512,64" "512,128,64") # Try smaller networks (e.g., "64", "64,32", "32")

EPOCHS=(10 25)  # Fewer epochs
HIDDENS=("64" "64,32")  # Smaller networks

ACC_STEPS=(0) #(0 2 4)

for bs in "${BATCH_SIZES[@]}"; do
  for lr in "${LRS[@]}"; do
    for ep in "${EPOCHS[@]}"; do
      for hd in "${HIDDENS[@]}"; do
        for acc in "${ACC_STEPS[@]}"; do
          #echo "Sweep: bs=$bs lr=$lr ep=$ep hd=$hd acc=$acc"
          SUFFIX="bs${bs}_lr${lr}_ep${ep}_hd${hd//,/_}_acc${acc}"
          LOGFILE="${OUT_DIR}/log_${SUFFIX}.txt"
          $PYTHON_CMD \
            --fm_model_name="$FM_MODEL" \
            --model_type="$MODEL_TYPE" \
            --data_path="$DATA_PATH" \
            --batch_size="$bs" \
            --meta="$META_PATH" \
            --phenotype_column="$PHENOTYPE_COLUMN" \
            --lr="$lr" \
            --epochs="$ep" \
            --hidden_dims="$hd" \
            --accumulation_steps="$acc" \
            --n_splits="$N_VAL_SPLITS" \
            --log_file="$LOGFILE" \
            --out_path="$OUT_DIR" \
            --do_cv \
            --cell_level

     
          newest=$(ls -t "${OUT_DIR}/cv_summary_"*.csv | head -n1)
          if [ ! -f "$SUMMARY_CSV" ]; then
            cat "$newest" > "$SUMMARY_CSV"
          else
            tail -n +2 "$newest" >> "$SUMMARY_CSV"
          fi

          # Clean up per-run summary
          rm -f "$newest"
        done
      done
    done
  done
done

echo "All sweeps launched."


####### TRAINING SCRIPT ENDS #########


echo "Archiving results..."
tar -cvf results_scFoundation_multiclass_r03x.tar results_scFoundation_r03x

echo "Job finished at $(date)"