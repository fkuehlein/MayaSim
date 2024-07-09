#!/bin/bash
#SBATCH --qos=short
#SBATCH --job-name=x11_mean
#SBATCH --account=copan
#SBATCH --output=x11_mean_%j.out
#SBATCH --error=x11_mean_%j.err
#SBATCH --chdir=/p/tmp/fritzku/MayaSim
#SBATCH --ntasks=64
#SBATCH --time=02:00:00

module load intel/oneAPI/2024.0.0
module load anaconda/2023.09
export OMP_NUM_THREADS=1

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ~/MayaSim/experiments/
mpirun -n $SLURM_NTASKS python x11_dynamical_regimes.py --mode 1
