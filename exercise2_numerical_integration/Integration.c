#include <stdio.h>
#include <math.h>
#include <omp.h>

/*
Log: Integration (subinter. size: 10 millions) with 1 threads: 0.631418, time: 0.473914
Log: Integration (subinter. size: 10 millions) with 2 threads: 0.631418, time: 0.250166
Log: Integration (subinter. size: 10 millions) with 4 threads: 0.631418, time: 0.125667
Log: Integration (subinter. size: 10 millions) with 8 threads: 0.631418, time: 0.063240
Log: Integration (subinter. size: 10 millions) with 16 threads: 0.631418, time: 0.032767
Log: Integration (subinter. size: 10 millions) with 32 threads: 0.631418, time: 0.030698

Log: Integration (subinter. size: 100 millions) with 1 threads: 0.631418, time: 4.717422
Log: Integration (subinter. size: 100 millions) with 2 threads: 0.631418, time: 2.492347
Log: Integration (subinter. size: 100 millions) with 4 threads: 0.631418, time: 1.250318
Log: Integration (subinter. size: 100 millions) with 8 threads: 0.631418, time: 0.625059
Log: Integration (subinter. size: 100 millions) with 16 threads: 0.631418, time: 0.314649
Log: Integration (subinter. size: 100 millions) with 32 threads: 0.631418, time: 0.170337

billion
Log: Integration (subinter. size: 1000 millions) with 1 threads: 0.631418, time: 47.189399
Log: Integration (subinter. size: 1000 millions) with 2 threads: 0.631418, time: 24.205269
Log: Integration (subinter. size: 1000 millions) with 4 threads: 0.631418, time: 10.511911
Log: Integration (subinter. size: 1000 millions) with 8 threads: 0.631418, time: 6.287754
Log: Integration (subinter. size: 1000 millions) with 16 threads: 0.631418, time: 3.144495
Log: Integration (subinter. size: 1000 millions) with 32 threads: 0.631418, time: 1.654046

*/

double f(double x)
{
    return sin(x * x);
}

double trapezoidal(double a, double b, int n)
{
    double h = (b - a) / n;
    double sum;
    #pragma omp parallel for reduction(+ : sum)
    for (int i = 1; i < n; i++)
    {
        double x = a + i * h;
        sum += f(x);
    }
    return h * (sum + (f(a) + f(h * n)) / 2);
}

int main()
{
    double a = 0.0;
    double b = 100.0;
    int n = 1000000000;

    double dt = omp_get_wtime();
    double result = trapezoidal(a, b, n);

    printf("Log: Integration (subinter. size: %d millions) with %d threads: %lf, time: %lf\n",
           n / 1000000, omp_get_max_threads(), result, omp_get_wtime() - dt);
    return 0;
}
