#!/bin/bash

#SBATCH --job-name=Opt_mix
#SBATCH --output=res.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=72:00:00
#SBATCH --mem=30000mb

module purge
module load jupyter/tensorflow/2023-10-10
. /pfs/data5/home/kit/mvm/px2030/code/tensorflow/bin/activate
ray start --head --port=6379

python opt_para_study.py

ray stop
deactivate

