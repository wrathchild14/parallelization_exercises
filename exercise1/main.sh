#!/bin/sh
#SBATCH --job-name=emb_p_1
#SBATCH --time=00:20:00
#SBATCH --output=main.txt
#SBATCH --reservation=fri

# divide the video in parts
srun --ntasks=1 ffmpeg -y -i bbb.mp4 -codec copy -f segment -segment_time 80 -segment_list parts.txt part-%d.mp4

# change output_parts.txt in correct format: file 'asdasd.mp4'
if [ ! -f output_parts.txt ]; then
    sed 's/^/file /' parts.txt > output_parts.txt
    sed -i "s/\([^ ]* \)\([^ ]*\)/\1'out-\2'/g" output_parts.txt
fi

sbatch --wait scale_video.sh

# concatinate the videos
srun --ntasks=1 ffmpeg -y -f concat -i output_parts.txt -c copy out-bbb.mp4

# clean up
rm *part*
rm *.txt
