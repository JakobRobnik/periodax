#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J quasars
#SBATCH -t 01:00:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# parameters of the script
mode='real'
temp='basic'

# load environment
module load python
conda activate quasar

# prepare the folder structure
python3 -m quasars.scratch_structure start

# run the analysis (split in batches because of the weird jax error)
for i in {0..17}
do
   start=$((i*2000))
   finish=$((start+2000))
   echo $start
   srun -n 128 -c 1 python -m quasars.run $start $finish 0.0 $mode $temp
done

# combine the results in a single file
python3 -m quasars.scratch_structure white


#python -m quasars.run 0 0 0.0 real basic
