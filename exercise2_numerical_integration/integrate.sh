#!/bin/bash
#SBATCH --job-name=integrate
#SBATCH --output=integrate.out
#SBATCH --time=00:01:00
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --reservation=fri

# compile
gcc --openmp -o integration_out Integration.c -lm

# run
srun --ntasks=1 integration_out

# clean
rm integration_out