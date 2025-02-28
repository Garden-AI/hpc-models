# These are the functions invoked by the bespoke AlphaFoldGarden.
# They are registered like gc_client.register_function(submit)

def submit(fasta_file_contents):

  def make_script(pdb_id):
    return f"""#!/bin/bash
#SBATCH --mem=240g
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1

#SBATCH --cpus-per-task=64    #assign all CPUs on node. Let ray decide distribution
#SBATCH --gpus-per-task=4     #assign all GPUs on node. Let ray decide distribution

#SBATCH --partition=gpuA40x4      # <- or one of: gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8

#SBATCH --account=bdao-delta-gpu

#SBATCH --job-name=protein_{pdb_id}_A40
#SBATCH --time=00:45:00      # hh:mm:ss for the job
#SBATCH -e protein_{pdb_id}_A40-err-%j.log
#SBATCH -o protein_{pdb_id}_A40-out-%j.log
#SBATCH --constraint="scratch"

### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpu-bind=closest     # <- or closest
#SBATCH --mail-user=willengler@uchicago.edu
#SBATCH --mail-type="BEGIN,END" # See sbatch or srun man pages for more email options

#to avoid mutually exclusive error
#set -x
unset SLURM_TRES_PER_TASK

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
conda deactivate
conda deactivate  # just making sure
module purge
module reset  # load the default Delta modules

module load anaconda3_gpu
module list

source /sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh
cd /scratch/bdao/parthpatel7173/APACE/remote_invocation

export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION=".75"
export XLA_PYTHON_CLIENT_ALLOCATOR="platform"

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${{nodes_array[0]}}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${{#ADDR[0]}} -gt 16 ]]; then
  head_node_ip=${{ADDR[1]}}
else
  head_node_ip=${{ADDR[0]}}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"


export head_node_ip
export port
export SLURM_GPUS_PER_TASK=4

srun -n 1 --gpus-per-task 4 --wait=10  \
singularity run --nv \
--bind /scratch/bdao/parthpatel7173/APACE/remote_invocation/inputs/:/mnt/fasta_inputs \
--bind /ime/hdd/rhaas/SUP-5301/database/uniref90:/mnt/uniref90_database_path \
--bind /ime/hdd/rhaas/SUP-5301/database/mgnify:/mnt/mgnify_database_path \
--bind /ime/hdd/rhaas/SUP-5301/database:/mnt/data_dir \
--bind /ime/hdd/rhaas/SUP-5301/database/pdb_mmcif/mmcif_files:/mnt/template_mmcif_dir \
--bind /ime/hdd/rhaas/SUP-5301/database/pdb_mmcif:/mnt/obsolete_pdbs_path \
--bind /ime/hdd/rhaas/SUP-5301/database/pdb70:/mnt/pdb70_database_path \
--bind /ime/hdd/rhaas/SUP-5301/database/uniref30:/mnt/uniref30_database_path \
--bind /ime/hdd/rhaas/SUP-5301/database/bfd:/mnt/bfd_database_path \
--bind /scratch/bdao/parthpatel7173/APACE/remote_invocation/outputs/{pdb_id}:/mnt/output \
./oct23-2024_image/apace-delta_latest.sif --fasta_paths=/mnt/fasta_inputs/{pdb_id}.fasta \
--uniref90_database_path=/mnt/uniref90_database_path/uniref90.fasta \
--mgnify_database_path=/mnt/mgnify_database_path/mgy_clusters_2022_05.fa \
--data_dir=/mnt/data_dir \
--template_mmcif_dir=/mnt/template_mmcif_dir \
--obsolete_pdbs_path=/mnt/obsolete_pdbs_path/obsolete.dat \
--pdb70_database_path=/mnt/pdb70_database_path/pdb70 \
--uniref30_database_path=/mnt/uniref30_database_path/UniRef30_2021_03 \
--bfd_database_path=/mnt/bfd_database_path/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
--output_dir=/mnt/output \
--max_template_date=2023-07-13 \
--db_preset=full_dbs \
--model_preset=monomer \
--benchmark=False \
--use_precomputed_msas=True \
--num_multimer_predictions_per_model=1 \
--models_to_relax=all \
--use_gpu_relax=True \
--logtostderr \
--perform_MD_only=False \
--use_amber=True \

echo "Job Completed"
"""

  import subprocess
  import os
  import sys

  root_dir = "/scratch/bdao/parthpatel7173/APACE/remote_invocation"

  # Get the PDB Id with some string munging
  pdb_header_first_portion = fasta_file_contents.split("\n")[0].split("|")[0]
  pdb_id = pdb_header_first_portion[1:].split("_")[0].lower()

  # Write the fasta file to disk in the right location.
  with open(f"{root_dir}/inputs/{pdb_id}.fasta", "w") as f:
    f.write(fasta_file_contents)
  # Make sure the output dir exists
  os.makedirs(f"{root_dir}/outputs/{pdb_id}", exist_ok=True)

  # Then generate the job script and write it to disk.
  script_contents = make_script(pdb_id)
  with open(f"{root_dir}/{pdb_id}_submit_script.sh", "w") as f:
    f.write(script_contents)

  # Then submit the job and return the response
  result = subprocess.Popen("/usr/bin/sbatch /scratch/bdao/parthpatel7173/APACE/faster_sub_script.sh",
                            shell=True,
                            universal_newlines=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

  output, error = result.communicate()
  if error:
    return output + error

  # output will be a string like "Submitted batch job 7001772"
  job_id = output.split()[-1]
  return {"job_id": job_id, "pdb_id": pdb_id}

def check_prediction_status(job_id):
  import subprocess
  result = subprocess.Popen(f"squeue -j {job_id}", shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

  output, error = result.communicate()

  if (str(job_id) in output):
    msg = "Job is queued or running\n"
    return msg + output
  else:
    msg = "Job has completed."
    return msg

def retrieve(pdb_id):
  import os

  root_dir = "/scratch/bdao/parthpatel7173/APACE/remote_invocation"

  output_dir = f"{root_dir}/outputs/{pdb_id}"
  file_path = f"{root_dir}/outputs/{pdb_id}/ranked_0_amber.pdb"
  if os.path.exists(file_path):
    with open(file_path, "r") as f:
      file_contents = f.read()
    return {
        "structure_file_contents": file_contents,
        "full_output_location": output_dir
    }
  else:
    return None