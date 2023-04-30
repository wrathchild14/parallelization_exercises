#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

#define SEED 42
#define SAMPLES 50000000

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
    double process_samples = SAMPLES / num_process;

    if (rank != 0)
    {
        double sample_start = process_samples * rank;
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
        count += process_samples;
        pi = ((double)count / SAMPLES) * 4.0;
        double end = MPI_Wtime();
        printf("PI = %f | computed in %.2f s\n", pi, end - start);
    }
    MPI_Finalize();
    return 0;
}
