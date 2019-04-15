#!/bin/bash
#SBATCH --qos=short
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --job-name=Maya_9_compute
#SBATCH --output=ms_x10_compute_out/%j.out
#SBATCH --error=ms_x10_compute_err/%j.err
#SBATCH --account=copan
#SBATCH --nodes=3
#SBATCH --tasks-per-node=7
#SBATCH --array=1-120

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

cd ../Experiments/
srun -n $SLURM_NTASKS python mayasim_X10_generate_trajectories.py 0 0 $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_MAX
