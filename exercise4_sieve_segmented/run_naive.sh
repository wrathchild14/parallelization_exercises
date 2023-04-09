#!/bin/sh
if [[ -z $1 ]];
then 
    threads=1
else
    threads=$1
fi
gcc -O2 sieve_naive.c --openmp -o sieve
srun --reservation=fri --cpus-per-task=$threads --constraint=AMD perf stat -B -e cache-references,cache-misses,cycles,stalled-cycles-backend,instructions,branches,branch-misses ./sieve 1000000000 
