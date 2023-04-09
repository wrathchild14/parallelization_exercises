// C program to count primes smaller than or equal to N
// compile: gcc -O2 sieve_naive.c --openmp -o sieve
// example run: ./sieve 20

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <omp.h>

void SieveOfEratosthenes(int n)
{
    //Initialize the array of flags for the primes
    unsigned char * primes = (unsigned char *)malloc((n+1) * sizeof(unsigned char));
    if (!primes) return;
    memset(primes, 1, (n+1));

    //Intialize array for the prime counts
    unsigned int * prime_cnts = (unsigned int *)calloc(omp_get_max_threads(), sizeof(unsigned int));
    double start = omp_get_wtime();
    //check for primes until sqrt(n)
    for (int p = 2; p * p <= n; p++)
    {

        //if flag is set then we encountered a prime number
        if (primes[p])
        {
            //cross out multiples of p grater than the square of p,
            //smaller have already been marked
            #pragma omp parallel for
            for (int i = p * p; i <= n; i += p)
                primes[i] = 0;
        }
    }
    
    unsigned int totalPrimes=0;
    //find and sum up all primes
    #pragma omp parallel
    {
        int id=omp_get_thread_num();
        #pragma omp for
        for (int p = 2; p <= n; p++) 
            if (primes[p])
                prime_cnts[id]++;
    
    }
    for(int i=0; i<omp_get_max_threads(); i++)
    {
        totalPrimes+=prime_cnts[i];
    }
    double stop=omp_get_wtime();
    printf("Total primes less or equal to %d: %d\n",n,totalPrimes);
    printf("Elapsed time: %.3f\n",stop-start);
    free(primes);
    free(prime_cnts);
}
 
// Driver Code
int main(int argc,char* argv[])
{
    if(argc<2){
        printf("Not enough arguments!\n");
        printf("Usage: sieve <N>!\n");
        return 1;
    }
    unsigned int N = atoi(argv[1]);

    SieveOfEratosthenes(N);
    return 0;
}