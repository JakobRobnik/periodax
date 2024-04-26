#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J quasars
#SBATCH -t 02:15:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# parameters of the script
mode='real'
temp=0
whichamp=0
#temp='randomized'

# load environment
module load python
conda activate quasar

# prepare the folder structure
python3 -m quasars.scratch_structure start $mode $temp $whichamp
for i in {0..8}
do
   start=$((i*3950))
   finish=$((start+3950))
   echo $start
   python -m quasars.runmult $start $finish
done
python3 -m quasars.scratch_structure finish $mode $temp $whichamp