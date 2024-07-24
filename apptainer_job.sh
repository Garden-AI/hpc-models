#!/bin/bash
#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpuA40x4
#SBATCH --account=bdao-delta-gpu
#SBATCH --job-name=pytorch_test
### GPU options ###
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1

module reset
module list

echo "Job is starting on $(hostname)"

apptainer exec --nv --bind environment.yml:/environment.yml --bind test_lib.py:/test_lib.py /sw/external/NGC/pytorch_22.08-py3.sif /bin/bash -c "
    source /opt/conda/bin/activate
    
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

    python test_lib.py
"
