#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J quasars
#SBATCH -t 00:10:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# parameters of the script
mode='inj'
temp=1 # to produce plot, run with 0 and with 1
whichamp=0

# load environment
module load python
conda activate quasar

# prepare the folder structure
python3 -m quasars.scratch_structure start $mode $temp $whichamp
python -m quasars.run_inj $temp
python3 -m quasars.scratch_structure finish $mode $temp $whichamp