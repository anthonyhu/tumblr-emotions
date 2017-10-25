#!/bin/bash
# Note anything after #SBATCH is a command
#
# Email you if job starts, completed or failed
#SBATCH --mail-user=ahu@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=week_trend
#Choose your partition depending on your requirements
#SBATCH --partition=medium
#SBATCH --ntasks=1
#SBATCH --nodes=1
# CPUs required for each job
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
# Memory per CPU in megabytes
#SBATCH --mem-per-cpu=32000
#SBATCH --array=0-0%1
#SBATCH --output=/data/localhost/not-backed-up/ahu/jobname_%A_%a.txt

# Now run your script.
python job_week.py

# Print environment variables:
echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

# Make a directory for output (.txt) and results (e.g. .jld, .json ,.mat) if it doesn't already exist
mkdir -p /data/ziz/not-backed-up/ahu/outputs/jobname_${SLURM_ARRAY_JOB_ID}
mkdir -p /data/ziz/not-backed-up/ahu/results/jobname_${SLURM_ARRAY_JOB_ID}

# Move experiment outputs & results to the directories made above
# These files are temporary and then moved to /outputs or /results
# Note that /data/localhost and /data/ziz contain the same files
mv /data/localhost/not-backed-up/ahu/jobname_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt /data/ziz/not-backed-up/ahu/outputs/jobname_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.txt
#mv /data/localhost/not-backed-up/ahu/jobname_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out /data/ziz/not-backed-up/ahu/results/jobname_${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}.out