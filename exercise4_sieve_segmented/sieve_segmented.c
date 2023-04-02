// C program to count primes smaller than or equal to N
// compile: gcc -O2 sieve_segmented.c --openmp -o sieve_segmented -lm
// example run: ./sieve_segmented 20

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

// Naive sieve time = 9.667

// RESULTS (averaged) for testing with: N = 1 billion, S (segments) = 500 thousand

#define SEGMENT_SIZE 500000

// Played with the segment size, found out the 500 thousand was
// overall the fastest for all thread numbers
/*
Elapsed time for 1 thread: 8.356 (SERIAL)
Elapsed time for 2 threads: 7.086  - Speedup: 1.179
Elapsed time for 4 threads: 3.562  - Speedup: 2.345
Elapsed time for 8 threads: 1.791  - Speedup: 4.665
Elapsed time for 16 threads: 0.927 - Speedup: 9.014
Elapsed time for 32 threads: 0.562 - Speedup: 14.865
*/

// Testing without "#pragma omp parallel for" on line 54
/* 
Elapsed time for 1 threads: 8.382 (SERIAL)
Elapsed time for 2 threads: 7.094  - Speedup: 1.181
Elapsed time for 4 threads: 2.635  - Speedup: 3.181
Elapsed time for 8 threads: 1.778  - Speedup: 4.714
Elapsed time for 16 threads: 0.897 - Speedup: 9.344
Elapsed time for 32 threads: 0.450 - Speedup: 18.626

We can see that with this change we start out the same with fewer threads,
but later on we can see a noticeable difference and speedup
*/
void SieveOfEratosthenes(int n)
{
    unsigned char *primes = (unsigned char *)malloc((n + 1) * sizeof(unsigned char));
    if (!primes) return;
    memset(primes, 1, (n + 1));

    unsigned int max_threads = omp_get_max_threads();
    unsigned int *prime_counts = (unsigned int *)calloc(max_threads, sizeof(unsigned int));

    double start = omp_get_wtime();

    unsigned int sqrt_n = sqrt(n);
    for (int p = 2; p <= sqrt_n; p++)
        if (primes[p])
            // #pragma omp parallel for
            for (int i = p * p; i <= sqrt_n; i += p)
                primes[i] = 0;

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i <= n; i += SEGMENT_SIZE)
    {
        // Segment start, end
        int start = i;
        unsigned int end = i + SEGMENT_SIZE - 1;
        if (end > n)
            end = n;

        unsigned char *segment_primes = (unsigned char *)malloc((SEGMENT_SIZE + 1) * sizeof(unsigned char));
        memset(segment_primes, 1, (SEGMENT_SIZE + 1));

        // Multiples of primes in first segment
        for (int p = 2; p <= sqrt_n; p++)
        {
            if (primes[p])
            {
                int start_multiple = ceil((double)start / p) * p;
                if (start_multiple < p * p)
                    start_multiple = p * p;
                for (int i = start_multiple; i <= end; i += p)
                    segment_primes[i - start] = 0;
            }
        }

        unsigned int prime_count = 0;
        for (int p = start; p <= end; p++)
            if (segment_primes[p - start])
                prime_count++;

        prime_counts[omp_get_thread_num()] += prime_count;
        free(segment_primes);
    }

    unsigned int total_primes = 0;
    for (int i = 0; i < max_threads; i++)
        total_primes += prime_counts[i];

    double stop = omp_get_wtime();

    printf("Total primes less than or equal to %d: %d\n", n, total_primes);
    printf("Elapsed time for %d threads: %.3f\n", max_threads, stop - start);

    free(primes);
    free(prime_counts);
}

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Not enough arguments!\n");
        printf("Usage: sieve <N>!\n");
        return 1;
    }
    unsigned int N = atoi(argv[1]);

    SieveOfEratosthenes(N);
    return 0;
}