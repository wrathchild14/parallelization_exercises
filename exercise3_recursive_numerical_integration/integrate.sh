#!/bin/bash
#SBATCH --job-name=quad_integrate
#SBATCH --output=quad_integrate.out
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --reservation=fri

export OMP_PLACES=cores
export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=16

# compile
gcc -O2 --openmp -o integration_out integration.c -lm

# run
srun --ntasks=1 integration_out

# clean
rm integration_out