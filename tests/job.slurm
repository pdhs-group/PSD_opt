#!/bin/bash

#SBATCH --job-name=Opt_mix
#SBATCH --output=res.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=72:00:00
#SBATCH --mem=30000mb

module purge
module load devel/python/3.12.3_gnu_13.3

python opt_para_study.py

