#!/bin/bash

#SBATCH --job-name=data
#SBATCH --output=res_data.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=30
#SBATCH --mem=40000mb

module purge
module load devel/python/3.11.7_intel_2021.4.0
export TMP_PATH=$TMPDIR

tar -C $TMPDIR/ -xzf $(ws_find MC_train)/mix_data.tar.gz
. /pfs/data5/home/kit/mvm/px2030/code/ray/bin/activate

python ../pypbe/utils/general_scripts/psd_opt_post.py

deactivate