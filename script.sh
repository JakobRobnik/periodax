#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J quasars
#SBATCH -t 00:30:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


module load python
conda activate quasar
python3 -m quasars.scratch_structure start
srun -n 128 -c 1 python -m simulations.roc 0.0
python3 -m quasars.scratch_structure amp0.0
python3 -m quasars.scratch_structure start
srun -n 128 -c 1 python -m simulations.roc 0.5
python3 -m quasars.scratch_structure amp0.5