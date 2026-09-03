#!/bin/bash

#SBATCH --job-name=salt_training
#SBATCH --output=/ptmp/viktorg/SALTImplementation/logs/salt_%j.out
#SBATCH --error=/ptmp/viktorg/SALTImplementation/logs/salt_%j.err

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G

#SBATCH --time=00:10:00

# Activate environment
eval "$(conda shell.bash hook)"
conda activate salt

# Update comet offline logs directory
COMET_BASE_DIR="${COMET_OFFLINE_DIRECTORY:-/ptmp/viktorg/SALTImplementation/comet_logs}"
export COMET_OFFLINE_DIRECTORY="${COMET_BASE_DIR}/${SLURM_JOB_NAME}_job_${SLURM_JOB_ID}"
mkdir -p "$COMET_OFFLINE_DIRECTORY"

echo "----- JOB INFO -----"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Base Config: $CONFIG_BASE_PATH"
echo "Model Config: $CONFIG_PATH"
echo "Comet Directory: $COMET_OFFLINE_DIRECTORY"
echo "--------------------"

# Execute training
salt fit -c "$CONFIG_BASE_PATH" -c "$CONFIG_PATH" --force
