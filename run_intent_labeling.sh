#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spot-vhmem

#SBATCH -o ./slurm/slurm.%j.out
#SBATCH -e ./slurm/slurm.%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=u14jp20@abdn.ac.uk

module load anaconda3
source activate master_venv

srun python intent_labeling.py