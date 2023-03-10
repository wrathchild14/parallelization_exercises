#!/bin/sh
#SBATCH --job-name=scale
#SBATCH --time=00:20:00
#SBATCH --output=ffmpeg3-%a.txt
#SBATCH --array=0-4
#SBATCH --reservation=fri

# scale down to half of the resolution
srun ffmpeg -y -i part-$SLURM_ARRAY_TASK_ID.mp4 -codec:a copy -filter:v scale=w=iw*1.5:h=ih*1.5 out-part-$SLURM_ARRAY_TASK_ID.mp4
