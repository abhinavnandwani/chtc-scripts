#!/bin/bash

# Step 1: Unpack scFoundation codebase
tar -xf scFoundation.tar

# Step 2: Unpack the donor dataset (creates MSSM/ directory)
DATA_TAR="MSSM_scFoundation_input.tar"
tar -xf "$DATA_TAR"

# Step 3: Copy label and pretrained checkpoint into model directory
#cp c02x_split_seed42.pkl scFoundation/model/
#cp scFoundation_pretrained.ckpt scFoundation/model/

cp scFoundation_v2_wb.py scFoundation/model/
# Step 4: Change into working directory
cd scFoundation/model || exit 1

# Step 5: Install wandb
pip install --user wandb

# Step 6: Run fine-tuning script using MSSM/ as the data path
python3 scFoundation_v2_wb.py --data_path "../../MSSM"

# Step 7: Package trained model directory
tar -cvf finetuned_c02x.tar models/finetuned_c02x

# Step 8: Move tarball to top-level directory for HTCondor transfer
cp finetuned_c02x.tar ../../..
