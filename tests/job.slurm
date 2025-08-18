#!/bin/bash

#SBATCH --job-name=Opt
#SBATCH --output=res.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=80
#SBATCH --time=72:00:00
#SBATCH --mem=80000mb

module purge
module load devel/python/3.11.7-gnu-14.2
export TMP_PATH=$TMPDIR
export STORAGE_PATH="$(ws_find ws)/Batch_Opt"
export TEST_GROUP="group1"
export PY_FILE="opt_Batch.py"

(
    SLEEP_TIME=$((72*3600 - 300))
    sleep $SLEEP_TIME

    echo "Time limit approaching, stopping main task..."

    pkill -f $PY_FILE

    if [ -d "$TMP_PATH/Ray_Tune" ]; then
        tar -czf $TMP_PATH/Ray_Tune_${TEST_GROUP}.tar.gz -C $TMP_PATH Ray_Tune
        rsync -av $TMP_PATH/Ray_Tune_${TEST_GROUP}.tar.gz $STORAGE_PATH/
    fi
) &

WATCHDOG_PID=$!

tar -C $TMP_PATH/ -xzf $STORAGE_PATH/data.tar.gz
mkdir $TMP_PATH/optframework
tar -C $TMP_PATH/optframework -xzf $STORAGE_PATH/optframework_vir.tar.gz
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
    tar -czf $TMP_PATH/Ray_Tune_${TEST_GROUP}.tar.gz -C $TMP_PATH Ray_Tune
    rsync -av $TMP_PATH/Ray_Tune_${TEST_GROUP}.tar.gz $STORAGE_PATH/
fi

deactivate



