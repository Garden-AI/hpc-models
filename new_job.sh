#!/bin/bash

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # Match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4
#SBATCH --time=00:10:00
#SBATCH --account=bdao-delta-gpu   # Match to a "Project" returned by the "accounts" command
#SBATCH --job-name=physnet_training
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

# Reset modules to a consistent state
module reset

# Load necessary modules
# module load anaconda3_gpu
module load cuda/11.8.0
module load gcc/11.4.0
module load openmpi/4.1.6
module load cudnn/8.9.0.131

# Get the current user's home directory
HOME_DIR=$HOME

# Define the environment path
ENV_PATH="$HOME_DIR/.conda/envs/physnet_env"

# Check if the environment exists
if [ ! -d "$ENV_PATH" ]; then
    echo "Environment does not exist. Creating environment from env.yml..."
    conda env create -f env.yml
else
    echo "Environment exists. Updating environment from env.yml..."
    conda env update --prefix $ENV_PATH --file env.yml --prune
fi

# Activate the environment
source activate $ENV_PATH

conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch
conda install pyg-lib -c pyg

# Check if conda activation was successful
if [ $? -ne 0 ]; then
  echo "Failed to activate conda environment."
  exit 1
fi

echo "Job is starting on $(hostname)"
echo "Current PATH: $PATH"

# Set the number of threads to match CPUs per task
export OMP_NUM_THREADS=16

# Find the full path of the Python executable
PYTHON_PATH=$(which python)
echo "Python path: $PYTHON_PATH"

# Check if the Python path is not empty
if [ -z "$PYTHON_PATH" ]; then
  echo "Python not found"
  exit 1
fi

# Run the Python script using the full path and handle potential errors
srun $PYTHON_PATH main.py > phys.out
if [ $? -ne 0 ]; then
  echo "srun command failed"
  exit 1
fi

# Deactivate the conda environment after the job is done
conda deactivate
