#!/bin/bash

#SBATCH --job-name=Opt
#SBATCH --output=res.txt
#SBATCH --nodes=8
#SBATCH --time=48:00:00
#SBATCH --mem=2000

module purge
module load devel/python/3.10.5_gnu_12.1

python opt_main.py
