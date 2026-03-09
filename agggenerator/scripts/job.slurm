#!/bin/bash

#SBATCH --job-name=LMC
#SBATCH --output=LMC.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=72:00:00
#SBATCH --mem=80000mb

module purge
module load devel/python/3.11.7-gnu-14.2
export TMP_PATH=$TMPDIR
export STORAGE_PATH="$(ws_find ws_me)/LMC_datasets"
export PY_FILE="simple_opt_mcpbe.py"

tar -C $TMP_PATH/ -xzf $STORAGE_PATH/data.tar.gz
mkdir $TMP_PATH/lmc
tar -C $TMP_PATH/lmc -xzf $STORAGE_PATH/lmc_vir.tar.gz
source $TMP_PATH/lmc/bin/activate

python $PY_FILE





