#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH -o ./slurm/slurm.%j.out
#SBATCH -e ./slurm/slurm.%j.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user=u14jp20@abdn.ac.uk

module load anaconda3
source activate master_venv

srun python belc_translation.py