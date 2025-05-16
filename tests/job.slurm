#!/bin/bash

#SBATCH --job-name=Opt
#SBATCH --output=res.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=30
#SBATCH --mem=60000mb

module purge
module load devel/python/3.11.7-gnu-14.2
export TMP_PATH=$TMPDIR
export STORAGE_PATH="$(ws_find ws)/Batch_Opt"

(
    SLEEP_TIME=$((72*3600 - 300))
    sleep $SLEEP_TIME

    echo "Time limit approaching, stopping main task..."

    pkill -f opt_Batch.py

    if [ -d "$TMPDIR/Ray_Tune" ]; then
        tar -czf $TMPDIR/Ray_Tune_${SLURM_JOB_ID}.tar.gz -C $TMPDIR Ray_Tune
        rsync -av $TMPDIR/Ray_Tune_${SLURM_JOB_ID}.tar.gz $(ws_find ws)/Batch_Opt/
    fi
) &

WATCHDOG_PID=$!

tar -C $TMPDIR/ -xzf $(ws_find ws)/Batch_Opt/data.tar.gz
tar -C $TMPDIR/ -xzf $(ws_find ws)/Batch_Opt/optframework_vir.tar.gz
source $TMPDIR/optframework/bin/activate

echo ">>> VIRTUAL_ENV: $VIRTUAL_ENV"
echo ">>> which python: $(which python)"
echo ">>> python --version: $(python --version)"
for pkg in ray numpy optframework; do
    echo ">>> Info for package: $pkg"
    pip show $pkg | grep -E 'Name|Version|Location'
done
echo "=============================="

python opt_Batch.py

kill $WATCHDOG_PID 2>/dev/null

if [ -d "$TMPDIR/Ray_Tune" ]; then
    tar -czf $TMPDIR/Ray_Tune_${SLURM_JOB_ID}.tar.gz -C $TMPDIR Ray_Tune
    rsync -av $TMPDIR/Ray_Tune_${SLURM_JOB_ID}.tar.gz $(ws_find ws)/Batch_Opt/
fi

deactivate



