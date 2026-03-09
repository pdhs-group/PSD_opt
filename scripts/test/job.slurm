#!/bin/bash

#SBATCH --job-name=MC02
#SBATCH --output=MCQ32.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --time=72:00:00
#SBATCH --mem=80000mb

module purge
module load devel/python/3.11.7-gnu-14.2
export TMP_PATH=$TMPDIR
export STORAGE_PATH="$(ws_find ws_me)/MCPBE_CB_Simon/WMCPBE"
export PY_FILE="simple_opt_mcpbe.py"
export TEST_GROUP="N100000_Q3_min_max2"
export RAY_memory_usage_threshold=0.995

(
    SLEEP_TIME=$((72*3600 - 300))
    sleep $SLEEP_TIME

    echo "Time limit approaching, stopping main task..."

    pkill -f $PY_FILE

    if [ -d "$TMP_PATH/Ray_Tune" ]; then
        tar -czf $TMP_PATH/Ray_Tune_${SLURM_JOB_ID}.tar.gz -C $TMP_PATH Ray_Tune
        rsync -av $TMP_PATH/Ray_Tune_${SLURM_JOB_ID}.tar.gz $STORAGE_PATH/
    fi
) &

WATCHDOG_PID=$!

tar -C $TMP_PATH/ -xzf $STORAGE_PATH/data.tar.gz
mkdir $TMP_PATH/optframework
tar -C $TMP_PATH/optframework -xzf $STORAGE_PATH/wmcpbe_vir.tar.gz
source $TMP_PATH/optframework/bin/activate

echo ">>> VIRTUAL_ENV: $VIRTUAL_ENV"
echo ">>> which python: $(which python)"
echo ">>> python --version: $(python --version)"
for pkg in ray numpy optframework; do
    echo ">>> Info for package: $pkg"
    pip show $pkg | grep -E 'Name|Version|Location'
done
echo "=============================="

python $PY_FILE

kill $WATCHDOG_PID 2>/dev/null

if [ -d "$TMP_PATH/Ray_Tune" ]; then
    tar -czf $TMP_PATH/Ray_Tune_${SLURM_JOB_ID}.tar.gz -C $TMP_PATH Ray_Tune
    rsync -av $TMP_PATH/Ray_Tune_${SLURM_JOB_ID}.tar.gz $STORAGE_PATH/
fi



