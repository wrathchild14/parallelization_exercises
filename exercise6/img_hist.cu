#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <device_launch_parameters.h>

#include "stb_image.h"
#include "stb_image_write.h"

#define GPU

void BasicCumHistCpu(unsigned int* hist_cum_normal);

void BlellochScanHistCpu(unsigned int* hist_cum, unsigned int* hist_min, bool print);

constexpr int HIST_CHANNELS = 3;
constexpr int BINS = 256;
constexpr int BLOCK_SIZE_VALUE = 256;

void GetHistogramCpu(const unsigned char* image_in,
    unsigned int* hist,
    const int width, const int height, const int channels_pp)
{
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
        {
            hist[image_in[(i * width + j) * channels_pp]]++; // RED
            hist[image_in[(i * width + j) * channels_pp + 1] + BINS]++; // GREEN
            hist[image_in[(i * width + j) * channels_pp + 2] + 2 * BINS]++; // BLUE
        }
}

__global__ void GetHistogramGpu(const unsigned char* image, unsigned int* histogram, const int width, const int height,
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
        atomicAdd(&histogram[g] + BINS, 1);
        atomicAdd(&histogram[b] + BINS * 2, 1);
    }
}

void PrintHistogram(const unsigned int* hist)
{
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < BINS; i++)
    {
        if (hist[i] > 0)
            printf("%dR\t%d\n", i, hist[i]);
        if (hist[i + BINS] > 0)
            printf("%dG\t%d\n", i, hist[i + BINS]);
        if (hist[i + 2 * BINS] > 0)
            printf("%dB\t%d\n", i, hist[i + 2 * BINS]);
    }
}

int main(const int argc, char** argv)
{
    char* image_file = "lena_small.png";
    unsigned int* hist = static_cast<unsigned int*>(calloc(HIST_CHANNELS * BINS, sizeof(unsigned int)));
    int width, height, cpp;
    unsigned char* image_in = stbi_load(image_file, &width, &height, &cpp, 0);
    printf("Log: image loaded with %d width, %d height, %d channels\n", width, height, cpp);

    if (image_in)
    {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

#ifdef GPU

        dim3 block_size(BLOCK_SIZE_VALUE);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
        printf("Log: block size %d %d, grid size %d %d\n", block_size.x, block_size.y, grid_size.x, grid_size.y);

        unsigned char* d_image;
        checkCudaErrors(cudaMalloc(&d_image, width * height * cpp * sizeof(unsigned char)));
        checkCudaErrors(
            cudaMemcpy(d_image, image_in, width * height * cpp * sizeof(unsigned char), cudaMemcpyHostToDevice));

        unsigned int* d_histogram;
        checkCudaErrors(cudaMalloc(&d_histogram, HIST_CHANNELS * BINS * sizeof(unsigned int)));

        GetHistogramGpu << <grid_size, block_size >> > (d_image, d_histogram, width, height, cpp);
        checkCudaErrors(
            cudaMemcpy(hist, d_histogram, HIST_CHANNELS * BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

#else

        histogram_cpu(image_in, hist, width, height, cpp);

#endif

        cudaEventRecord(stop);

        // Wait for the event to finish
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);


        // Histogram equalization
        auto hist_cum_normal = static_cast<unsigned int*>(malloc(HIST_CHANNELS * BINS * sizeof(unsigned int)));
        memcpy(hist_cum_normal, hist, HIST_CHANNELS * BINS * sizeof(unsigned int));
        BasicCumHistCpu(hist_cum_normal);


        // Init values for Blelloch inclusive scan
        auto hist_cum = static_cast<unsigned int*>(malloc(HIST_CHANNELS * BINS * sizeof(unsigned int)));
        memcpy(hist_cum, hist, HIST_CHANNELS * BINS * sizeof(unsigned int));

        auto hist_mins = static_cast<unsigned int*>(malloc(HIST_CHANNELS * sizeof(unsigned int)));
        for (unsigned int i = 0; i < HIST_CHANNELS; i++) {
            hist_mins[i] = 99999;
        }

        BlellochScanHistCpu(hist_cum, hist_mins, false);

        // Out-image should be 4 cpps
        auto image_out = static_cast<unsigned char*>(malloc(width * height * 4 * sizeof(unsigned char)));
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
            {
                const int index = (i * width + j) * 4;
                int r = image_in[index];
                int g = image_in[index + 1];
                int b = image_in[index + 2];

                image_out[index] = round(((static_cast<double>(hist_cum[r] - hist_mins[0])) / static_cast<double>((height * width - hist_mins[0]))) * (BINS - 1));
                image_out[index + 1] = round(((static_cast<double>(hist_cum[g + BINS] - hist_mins[1])) / static_cast<double>((height * width - hist_mins[1]))) * (BINS - 1));
                image_out[index + 2] = round(((static_cast<double>(hist_cum[b + BINS * 2] - hist_mins[2])) / static_cast<double>((height * width - hist_mins[2]))) * (BINS - 1));
                image_out[index + 3] = image_in[index + 3]; // Keep the A value
            }
        }
        stbi_write_png("lena_result.png", width, height, 4, image_out, width * 4);

        free(hist_cum);
        free(image_out);

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

void BlellochScanHistCpu(unsigned int* hist_cum, unsigned int* hist_min, bool print)
{
    for (int i = 1; i < BINS; i *= 2)
    {
        for (int j = 0; j < BINS; j += i * 2)
        {
            int red_value = hist_cum[j + i - 1];
            hist_cum[j + i * 2 - 1] += red_value;
            if (red_value > 0 && hist_min[0] == 99999) {
                hist_min[0] = red_value;
            }

            int green_value = hist_cum[j + i - 1 + BINS];
            hist_cum[j + i * 2 - 1 + BINS] += green_value;
            if (green_value > 0 && hist_min[1] == 99999) {
                hist_min[1] = green_value;
            }

            int blue_value = hist_cum[j + i - 1 + BINS * 2];
            hist_cum[j + i * 2 - 1 + BINS * 2] += blue_value;
            if (blue_value > 0 && hist_min[2] == 99999) {
                hist_min[2] = blue_value;
            }
        }
    }

    hist_cum[BINS - 1] = 0;
    hist_cum[BINS * 2 - 1] = 0;
    hist_cum[BINS * 3 - 1] = 0;

    for (int i = BINS / 2; i >= 1; i /= 2)
    {
        for (int j = 0; j < BINS; j += i * 2)
        {
            unsigned int temp;
            temp = hist_cum[j + i - 1];
            hist_cum[j + i - 1] = hist_cum[j + i * 2 - 1];
            hist_cum[j + i * 2 - 1] += temp;

            temp = hist_cum[j + i - 1 + BINS];
            hist_cum[j + i - 1 + BINS] = hist_cum[j + i * 2 - 1 + BINS];
            hist_cum[j + i * 2 - 1 + BINS] += temp;

            temp = hist_cum[j + i - 1 + BINS * 2];
            hist_cum[j + i - 1 + BINS * 2] = hist_cum[j + i * 2 - 1 + BINS * 2];
            hist_cum[j + i * 2 - 1 + BINS * 2] += temp;
        }
    }
    printf("Log: min values: r%d g%d b%d\n", hist_min[0], hist_min[1], hist_min[2]);
    if (print) {
        PrintHistogram(hist_cum);
    }
}

void BasicCumHistCpu(unsigned int* hist_cum_normal)
{
    int r_sum = 0;
    int g_sum = 0;
    int b_sum = 0;
    for (int i = 0; i < BINS; i++)
    {
        int red = hist_cum_normal[i];
        r_sum += red;
        hist_cum_normal[i] = r_sum;

        int green = hist_cum_normal[i + BINS];
        g_sum += green;
        hist_cum_normal[i + BINS] = g_sum;

        int blue = hist_cum_normal[i + BINS * 2];
        b_sum += blue;
        hist_cum_normal[i + BINS * 2] = b_sum;
    }
}
