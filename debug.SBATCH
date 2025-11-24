#!/bin/bash
# Usage: sbatch run.SBATCH <filename>

#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --mem=32g
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="l40s"
#SBATCH --job-name=zoology-debug
#SBATCH --output=slurm/logs/%j.out

module purge
module load "anaconda3/2025.06"

mkdir -p slurm/logs
mkdir -p slurm/scripts

scontrol write batch_script $SLURM_JOB_ID "slurm/scripts/$SLURM_JOB_ID-$SLURM_JOB_NAME.sh"

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK;
export MASTER_ADDR=$(hostname)
export MASTER_PORT=12345
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID


source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh;
export PATH_TO_ENV=/scratch/sl12886/tts_icl/zoology/penv
source activate /scratch/sl12886/tts_icl/zoology/penv
export PATH=$PATH_TO_ENV/bin:$PATH;
python -m zoology.launch zoology/experiments/gisting/configs.py

