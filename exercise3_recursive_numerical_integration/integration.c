// adaptive integration
// compile: gcc -O2 quad.c --openmp -lm -o quad

/*
RESULTS:

SERIAL
    Result: 0.631418 Time: 19.340

First I was trying the parallel algorithm with dynamic and static.
So, I switched with guided and got the best results


SEGMENTS = 1000 - wrong results
    Log: Threads 1 - Segments 1000 - Result 0.534862 Time: 0.611
    Log: Threads 2 - Segments 1000 - Result 0.534862 Time: 0.319
    Log: Threads 4 - Segments 1000 - Result 0.534862 Time: 0.160
    Log: Threads 8 - Segments 1000 - Result 0.534862 Time: 0.081
    Log: Threads 16 - Segments 1000 - Result 0.534862 Time: 0.042
    speedup: 14.547619047619046

SEGMENTS = 10000
    Log: Threads 1 - Segments 10000 - Result 0.631418 Time: 0.192
    Log: Threads 2 - Segments 10000 - Result 0.631418 Time: 0.100
    Log: Threads 4 - Segments 10000 - Result 0.631418 Time: 0.051
    Log: Threads 8 - Segments 10000 - Result 0.631418 Time: 0.024
    Log: Threads 16 - Segments 10000 - Result 0.631418 Time: 0.013
    Log: Threads 32 - Segments 10000 - Result 0.631418 Time: 0.008
    speedup: 24.0

SEGMENTS = 100000
    Log: Threads 1 - Segments 100000 - Result 0.631417 Time: 0.037
    Log: Threads 2 - Segments 100000 - Result 0.631417 Time: 0.018
    Log: Threads 4 - Segments 100000 - Result 0.631417 Time: 0.010
    Log: Threads 8 - Segments 100000 - Result 0.631417 Time: 0.005
    Log: Threads 16 - Segments 100000 - Result 0.631417 Time: 0.004
    Log: Threads 32 - Segments 100000 - Result 0.631417 Time: 0.005
    speedup: 7.3999999999999995

SEGMENTS = 1 million
    Log: Threads 1 - Segments 1000000 - Result 0.631418 Time: 0.325
    Log: Threads 4 - Segments 1000000 - Result 0.631418 Time: 0.082
    Log: Threads 8 - Segments 1000000 - Result 0.631418 Time: 0.042
    Log: Threads 16 - Segments 1000000 - Result 0.631418 Time: 0.022
    Log: Threads 32 - Segments 1000000 - Result 0.631418 Time: 0.013
    speedup: 25.000000000000004

SEGMENTS = 10 million
    Log: Threads 1 - Segments 10000000 - Result 0.631418 Time: 2.801
    Log: Threads 2 - Segments 10000000 - Result 0.631418 Time: 1.630
    Log: Threads 4 - Segments 10000000 - Result 0.631418 Time: 0.815
    Log: Threads 8 - Segments 10000000 - Result 0.631418 Time: 0.407
    Log: Threads 16 - Segments 10000000 - Result 0.631418 Time: 0.205
    Log: Threads 32 - Segments 10000000 - Result 0.631418 Time: 0.105
    speedup: 26.676190476190477

SEGMENTS = 100 million
    Log: Threads 4 - Segments 100000000 - Result 0.631418 Time: 8.269
    Log: Threads 8 - Segments 100000000 - Result 0.631418 Time: 4.061
    Log: Threads 16 - Segments 100000000 - Result 0.631418 Time: 2.030
    Log: Threads 32 - Segments 100000000 - Result 0.631418 Time: 1.021

SEGMENTS = 1 billion
    Log: Threads 16 - Segments 1000000000 - Result 0.631418 Time: 15.739
    Log: Threads 32 - Segments 1000000000 - Result 0.631418 Time: 8.651


I would say the 10000 segments are the sweetspot as it had the most speedup and correct result.
*/

#include <stdio.h>
#include <math.h>
#include <omp.h>

#define TOL 1e-6

// function to integrate
double func(double x)
{
    return sin(x * x);
}

// integration function with arguments
// function to integrate
// lower boundary
// upper boundary
// error tolerance
double quad(double (*f)(double), double lower, double upper, double tol)

{
    double quad_res;    // result
    double h;           // interval length
    double middle;      // middle point
    double quad_coarse; // coarse approximation
    double quad_fine;   // fine approximation (two trapezoids)
    double quad_lower;  // result on the lower interval
    double quad_upper;  // result on the upper interval
    double eps;         // difference

    h = upper - lower;
    middle = (lower + upper) / 2;

    // compute the integral using both approximations
    quad_coarse = h * (f(lower) + f(upper)) / 2.0;
    quad_fine = h / 2 * (f(lower) + f(middle)) / 2.0 + h / 2 * (f(middle) + f(upper)) / 2.0;
    eps = fabs(quad_coarse - quad_fine);

    // if not inside acceptable tolerance, split and repeat
    if (eps > tol)
    {
        quad_lower = quad(f, lower, middle, tol / 2);
        quad_upper = quad(f, middle, upper, tol / 2);
        quad_res = quad_lower + quad_upper;
    }
    else
        quad_res = quad_fine;

    return quad_res;
}

int main(int argc, char *argv[])
{
    int max_threads = omp_get_max_threads();
    int segments = 100000;
    double dt = omp_get_time();

    double a = 0.0, b = 100.0;
    double sub_intervals = (b - a) / segments;
    double quadrature_sum = 0;

    #pragma omp parallel
    {
        double quadrature_sum_sub = 0;
        #pragma omp for schedule(guided)
        for (int i = 0; i < segments; i++)
        {
            double a_sub = i * sub_intervals;
            double b_sub = (i + 1) * sub_intervals;
            quadrature_sum_sub += quad(func, a_sub, b_sub, TOL);
        }

        #pragma omp atomic
        quadrature_sum += quadrature_sum_sub;
    }

    dt = omp_get_time() - dt;
    printf("Log: Threads %d - Segments %d - Result %lf Time: %.3lf\n", max_threads, segments, quadrature_sum, dt);

    return 0;
}
