#!/bin/bash

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16     # Match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4
#SBATCH --time=00:30:00
#SBATCH --account=bdao-delta-gpu   # Match to a "Project" returned by the "accounts" command
#SBATCH --job-name=physnet_training
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

# Reset modules to a consistent state
module reset

# Load necessary modules
module load anaconda3_gpu  # 加载Anaconda模块
module load cuda/11.8.0
module load gcc/11.4.0
module load openmpi/4.1.6

echo "Job is starting on $(hostname)"
echo "Current PATH: $PATH"

# Set the number of threads to match CPUs per task
export OMP_NUM_THREADS=16

# Activate the Anaconda environment
source activate myenv  # 替换为你的Anaconda环境名称

# Find the full path of the Python executable
PYTHON_PATH=$(which python)
echo "Python path: $PYTHON_PATH"

# Check if the Python path is not empty
if [ -z "$PYTHON_PATH" ]; then
  echo "Python not found"
  exit 1
fi

# Run the Python script using the full path and handle potential errors
srun python /u/jisenli2/ondemand/hpc/apptainer_hpc/hpc-models/Apptainer/scripts/apptainer.py > myjobs33.out
if [ $? -ne 0 ]; then
  echo "srun command failed"
  exit 1
fi
