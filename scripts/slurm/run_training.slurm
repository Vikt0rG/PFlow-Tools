#!/bin/bash

#SBATCH --job-name=salt_training
#SBATCH --output=/ptmp/viktorg/SALTImplementation/logs/salt_%j.out
#SBATCH --error=/ptmp/viktorg/SALTImplementation/logs/salt_%j.err

#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

#SBATCH --time=02:00:00

# Activate environment
eval "$(conda shell.bash hook)"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-salt}"
conda activate "$CONDA_ENV_NAME"

# Update comet offline logs directory
COMET_BASE_DIR="${COMET_OFFLINE_DIRECTORY:-/ptmp/viktorg/SALTImplementation/comet_logs}"
export COMET_OFFLINE_DIRECTORY="${COMET_BASE_DIR}/${SLURM_JOB_NAME}_job_${SLURM_JOB_ID}"
mkdir -p "$COMET_OFFLINE_DIRECTORY"

echo "----- JOB INFO -----"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Model Config Flags: $CONFIG_ARGS"
echo "Comet Directory: $COMET_OFFLINE_DIRECTORY"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "--------------------"

salt fit $CONFIG_ARGS --force