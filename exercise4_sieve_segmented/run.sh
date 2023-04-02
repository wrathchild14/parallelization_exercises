#!/bin/sh
#SBATCH --job-name=s_sieve
#SBATCH --output=sieve_segmented.out
#SBATCH --time=00:00:30
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --reservation=fri
#SBATCH --constraint=AMD

gcc -O2 sieve_segmented.c --openmp -o sieve_segmented -lm

perf stat -B -e cache-references,cache-misses,cycles,stalled-cycles-backend,instructions,branches,branch-misses ./sieve_segmented 1000000000

rm sieve_segmented