#!/bin/bash
#SBATCH --qos=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=MayaSim_x11
#SBATCH --output=ms_x8_%j.out
#SBATCH --error=ms_x8_%j.err
#SBATCH --account=copan
#SBATCH --nodes=4
#SBATCH --tasks-per-node=16

module load compiler/intel/16.0.0
module load hpc/2015 anaconda/2.3.0
export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so
export OMP_NUM_THREADS=1

source activate mayasim

##################
echo "------------------------------------------------------------"
echo "SLURM JOB ID: $SLURM_JOBID"
echo "$SLURM_NTASKS tasks"
echo "------------------------------------------------------------"

cd ./experiments/
srun -n $SLURM_NTASKS python x11_dynamical_regimes.py --testing --mode=0
