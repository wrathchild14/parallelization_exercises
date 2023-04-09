#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"

/*
 * README
 * Image is 4 channels, histogram is 3 channels
 * Main development was done on locally on my PC with GPU: NVIDIA GeForce RTX 3060 and these are the following results:
 * Time on CPU: 144.377 milliseconds
 *
 * Time on GPU with 32 block size and grid_size.x 253, grid_size.y 4048: 36.529 milliseconds
 * Time on GPU with 64 block size and grid_size.x 127, grid_size.y 4048: 31.218 milliseconds
 * Time on GPU with 128 block size and grid_size.x 64, grid_size.y 4048: 31.345 milliseconds
 * Time on GPU with 256 block size and grid_size.x 32, grid_size.y 4048: 31.540 milliseconds
 * Time on GPU with 512 block size and grid_size.x 16, grid_size.y 4048: 34.421 milliseconds
 * Time on GPU with 1024 block size and grid_size.x 8, grid_size.y 4048: 35.507 milliseconds
 * Best of these times were the ones with [64, 128, 256] block sizes with a speedup around 4.6
 * 
 */

#define GPU

constexpr int bins = 256;
constexpr int block_size_value = 1024;
constexpr int histogram_channels = 3;
unsigned int* hist;

void histogram_cpu(const unsigned char* image_in,
                   unsigned int* hist,
                   const int width, const int height, const int channels_pp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            hist[image_in[(i * width + j) * channels_pp]]++; // RED
            hist[image_in[(i * width + j) * channels_pp + 1] + bins]++; // GREEN
            hist[image_in[(i * width + j) * channels_pp + 2] + 2 * bins]++; // BLUE
        }
}

__global__ void histogram_gpu(const unsigned char* image, unsigned int* histogram, const int width, const int height,
                              const int channels_pp)
{
    const int x = threadIdx.x + blockIdx.x * blockDim.x;
    const int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x < width && y < height)
    {
        const int index = (y * width + x) * channels_pp;
        const int r = image[index];
        const int g = image[index + 1];
        const int b = image[index + 2];

        atomicAdd(&histogram[r], 1);
        atomicAdd(&histogram[g] + bins, 1);
        atomicAdd(&histogram[b] + bins * 2, 1);
    }
}

void print_histogram(const unsigned int* hist)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < bins; i++)
    {
        if (hist[i] > 0)
            printf("%dR\t%d\n", i, hist[i]);
        if (hist[i + bins] > 0)
            printf("%dG\t%d\n", i, hist[i + bins]);
        if (hist[i + 2 * bins] > 0)
            printf("%dB\t%d\n", i, hist[i + 2 * bins]);
    }
}


int main(const int argc, char** argv)
{
    char* image_file;

    if (argc > 1)
    {
        image_file = argv[1];
    }
    else
    {
        fprintf(stderr, "Not enough arguments\n");
        fprintf(stderr, "Usage: %s <IMAGE_PATH>\n", argv[0]);
        exit(1);
    }

    hist = static_cast<unsigned int*>(calloc(histogram_channels * bins, sizeof(unsigned int)));

    int width, height, cpp;
    unsigned char* image_in = stbi_load(image_file, &width, &height, &cpp, 0);
    printf("Log: image loaded with %d width, %d height, %d channels\n", width, height, cpp);

    if (image_in)
    {
        // Compute and print the histogram
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

#ifdef GPU

        dim3 block_size(block_size_value);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
        printf("Log: block size %d %d, grid size %d %d\n", block_size.x, block_size.y, grid_size.x, grid_size.y);

        unsigned char* d_image;
        checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));
        checkCudaErrors(
            cudaMemcpy(d_image, image_in, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned int* d_histogram;
        checkCudaErrors(cudaMalloc(&d_histogram, histogram_channels * bins * sizeof(unsigned int)));

        histogram_gpu<<<grid_size, block_size>>>(d_image, d_histogram, width, height, cpp);
        checkCudaErrors(
            cudaMemcpy(hist, d_histogram, histogram_channels * bins * sizeof(unsigned int), cudaMemcpyDeviceToHost));

#else

        histogram_cpu(image_in, hist, width, height, cpp);

#endif

        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // print_histogram(hist);

#ifdef GPU
        printf("Time on GPU with %d block size and grid_size.x %d, grid_size.y %d: %0.3f milliseconds \n",
               block_size.x, grid_size.x, grid_size.y, milliseconds);

        cudaFree(d_image);
        cudaFree(d_histogram);
#else
        printf("Time on CPU: %0.3f milliseconds \n", milliseconds);
#endif
    }
    else
    {
        fprintf(stderr, "Error loading image %s!\n", image_file);
    }

    return 0;
}
