#!/bin/sh
module load OpenMPI/4.1.0-GCC-10.2.0 
mpicc pi_mpi.c -o pi_mpi -lm

processes=(1 2 4 8 16 32)
nodes=(1 2)

for n in "${nodes[@]}"
do
  for p in "${processes[@]}"
  do
    srun --reservation=fri --mpi=pmix -n $p -N $n ./pi_mpi
    echo "Nodes: $n, Processes: $p"
  done
done

rm pi_mpi