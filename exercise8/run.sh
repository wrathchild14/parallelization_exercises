#!/bin/sh
module load OpenMPI/4.1.0-GCC-10.2.0 
mpicc pi_mpi.c -o pi_mpi -lm
# -n# - processes -N# - nodes
srun --reservation=fri --mpi=pmix -n8 -N2 ./pi_mpi
rm pi_mpi