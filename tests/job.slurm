#!/bin/bash

#SBATCH --job-name=Opt
#SBATCH --output=res.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=72:00:00
#SBATCH --mem=40000mb

module purge
module load devel/python/3.11.7_intel_2021.4.0
export TMP_PATH=$TMPDIR

tar -C $TMPDIR/ -xzf $(ws_find MC_train)/mix_data_shear2.tar.gz
tar -C $TMPDIR/ -xzf $(ws_find MC_train)/tensorflow_env2.tar.gz
. $TMPDIR/tensorflow/bin/activate

python opt_para_study.py

deactivate

