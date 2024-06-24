#!/bin/bash

#SBATCH -N 1
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -J quasars
#SBATCH -t 01:45:00
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=jakob_robnik@berkeley.edu

# parameters of the script #S-B-ATCH --array=0-9

mode='real'
temp=1000 #$SLURM_ARRAY_TASK_ID 
whichamp=0

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
   python -m quasars.run $start $finish $mode $temp $whichamp
done
python3 -m quasars.scratch_structure finish $mode $temp $whichamp