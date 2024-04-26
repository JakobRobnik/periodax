#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J quasars
#SBATCH -t 02:30:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


# parameters of the script
mode='sim'
temp='basic'
whichamp=4
#temp='randomized'

# load environment
module load python
conda activate periodax

# prepare the folder structure
python3 -m quasars.scratch_structure start $mode $temp $whichamp

# run the analysis (split in batches because of the weird jax error)
for i in {0..17}
do
   start=$((i*2000))
   finish=$((start+2000))
   echo $start
   srun -n 128 -c 1 python -m quasars.run $start $finish $mode $temp $whichamp
done

# combine the results in a single file
python3 -m quasars.scratch_structure finish $mode $temp $whichamp