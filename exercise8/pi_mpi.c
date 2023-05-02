#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define SEED 42
#define SAMPLES 5000000

/* Readme:
Samples = 5000000

PI = 3.142030 | computed in 0.57 s
Nodes: 1, Processes: 2
PI = 3.141841 | computed in 0.21 s
Nodes: 1, Processes: 4
PI = 3.143234 | computed in 0.10 s
Nodes: 1, Processes: 8
PI = 3.143058 | computed in 0.04 s
Nodes: 1, Processes: 16
PI = 3.142443 | computed in 0.04 s
Nodes: 1, Processes: 32

PI = 3.142030 | computed in 0.52 s
Nodes: 2, Processes: 2
PI = 3.141841 | computed in 0.19 s
Nodes: 2, Processes: 4
PI = 3.143234 | computed in 0.09 s
Nodes: 2, Processes: 8
PI = 3.143058 | computed in 0.05 s
Nodes: 2, Processes: 16
PI = 3.142443 | computed in 0.06 s
Nodes: 2, Processes: 32
*/

int main(int argc, char *argv[])
{
    int rank;
    int num_process;
    int source;
    int tag = 0;
    char message_buffer[100];
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_process);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double start = MPI_Wtime();
    double count = 0;
    double x, y, z, pi;
    double process_samples = SAMPLES / (num_process - 1);

    if (rank != 0)
    {
        srand(rank + SEED);
        int local_count = 0;
        for (int i = 0; i < process_samples; i++)
        {
            x = (double)rand() / (double)RAND_MAX;
            y = (double)rand() / (double)RAND_MAX;
            z = sqrt((x * x) + (y * y));

            if (z <= 1.0)
            {
                local_count++;
            }
        }

        sprintf(message_buffer, "%d", local_count);
        MPI_Send(message_buffer, (int)strlen(message_buffer) + 1, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
    }
    else
    {
        for (source = 1; source < num_process; source++)
        {
            MPI_Recv(message_buffer, 100, MPI_CHAR, source, tag, MPI_COMM_WORLD, &status);
            int local_count = atoi(message_buffer);
            count += local_count;
        }
        pi = ((double)count / SAMPLES) * 4.0;
        double end = MPI_Wtime();
        printf("PI = %f | computed in %.2f s\n", pi, end - start);
    }
    MPI_Finalize();
    return 0;
}
