#!/bin/bash
# Usage: sbatch run.SBATCH <filename>

#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --mem=64g
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --constraint="l40s"
#SBATCH --job-name=gist_vs_attn
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
export N_NODES=$SLURM_NNODES
export RANK=$SLURM_PROCID


source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh;
export PATH_TO_ENV=/scratch/sl12886/tts_icl/zoology/penv
source activate /scratch/sl12886/tts_icl/zoology/penv
export PATH=$PATH_TO_ENV/bin:$PATH;

# python -m zoology.launch zoology/experiments/gisting/configs_surjective.py -p
# python -m zoology.launch zoology/experiments/gisting/configs_mqgar.py
# python -m zoology.launch zoology/experiments/gisting/configs_mvqar.py -p
# python -m zoology.launch zoology/experiments/gisting/configs_inf_pairs.py -p
# python -m zoology.launch zoology/experiments/gisting/configs_inf_shots.py -p
# python -m zoology.launch zoology/experiments/gisting/configs_gistsa_vs_gist.py -p
# python -m zoology.launch zoology/experiments/gisting/configs_selfproto.py -p
python -m zoology.launch zoology/experiments/gisting/configs_gistsa_vs_attn.py -p
