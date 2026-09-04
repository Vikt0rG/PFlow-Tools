#!/bin/bash

#SBATCH --job-name=salt_testing
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
conda activate $CONDA_ENV_NAME

echo "----- JOB INFO -----"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Model Config: $CONFIG_PATH"
echo "Test file: $TEST_FILE_PATH"
echo "Devices: $TRAINER_DEVICES"
echo "Conda Environment: $CONDA_ENV_NAME"
echo "--------------------"

# Execute testing
salt test \
    -c "$CONFIG_PATH" \
    --data.test_file "$TEST_FILE_PATH" \
    --trainer.devices="$TRAINER_DEVICES"