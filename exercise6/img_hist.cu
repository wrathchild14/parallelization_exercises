#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#include "helper_cuda.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <device_launch_parameters.h>

#include "stb_image.h"
#include "stb_image_write.h"

// README
/*

On my PC (RTX 3060):
with 1024:512 res image:
	Time: 58.624 milliseconds
	Time: 3.993 milliseconds
	Speedup: 14.6816929626847
with 8096:4048 res image:
	Time: 3387.272 milliseconds
	Time: 84.465 milliseconds
	Speedup: 40.102669744864734

On NSC:
with 1024:512 res image:
	CPU Time: 32.036 milliseconds
	GPU Time: 3.588 milliseconds
	Speedup: 8.928651059085842
with 8096:4048 res image:
	CPU Time: 1870.367 milliseconds
	GPU Time: 77.086 milliseconds
	Speedup: 24.26338115870586

So first it looks suprising because of the speedups but the GPU times are similar.
The interesting thing is that on my processor it takes more time to execute and that is the reason why the speedup is so large.
*/

#define GPU

void GetHistogramCpu(const unsigned char* image_in, unsigned int* hist, const int width, const int height,
                     const int channels_pp);
void PrintHistogram(const unsigned int* hist);
void BasicCumHistCpu(unsigned int* hist_cum_normal);
void BlellochScanHistCpu(unsigned int* hist_cum, unsigned int* hist_min, bool print);
void BlellochScanHistGpu(unsigned int* hist_cum, unsigned int* hist_mins, bool print);
void EqulizeImageCpu(int height, int width, unsigned char* image_in, unsigned char* image_out, unsigned int* hist_cum,
                     unsigned int* hist_mins);

constexpr int HIST_CHANNELS = 3;
constexpr int BINS = 256;
constexpr int BLOCK_SIZE_VALUE = 256;

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

__global__ void EqualizeImageGpu(int height, int width, unsigned char* image_in, unsigned char* image_out,
                                 unsigned int* hist_cum, unsigned int* hist_mins)
{
	const int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
	const int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

	if (tid_x < width && tid_y < height)
	{
		const int index = (tid_y * width + tid_x) * 4;
		int r = image_in[index];
		int g = image_in[index + 1];
		int b = image_in[index + 2];

		image_out[index] = round(
			((static_cast<double>(hist_cum[r] - hist_mins[0])) / static_cast<double>((height * width - hist_mins[0]))) *
			(BINS - 1));
		image_out[index + 1] = round(
			((static_cast<double>(hist_cum[g + BINS] - hist_mins[1])) / static_cast<double>((height * width - hist_mins[
				1]))) * (BINS - 1));
		image_out[index + 2] = round(
			((static_cast<double>(hist_cum[b + BINS * 2] - hist_mins[2])) / static_cast<double>((height * width -
				hist_mins[2]))) * (BINS - 1));
		image_out[index + 3] = image_in[index + 3]; // Keep the A value
	}
}

__global__ void BlellochScanHistKernel(unsigned int* hist_cum, unsigned int* hist_min)
{
	for (int i = 1; i < BINS; i *= 2)
	{
		for (int j = 0; j < BINS; j += i * 2)
		{
			int red_value = hist_cum[j + i - 1];
			hist_cum[j + i * 2 - 1] += red_value;
			if (red_value > 0 && hist_min[0] == INT_MAX)
			{
				hist_min[0] = red_value;
			}

			int green_value = hist_cum[j + i - 1 + BINS];
			hist_cum[j + i * 2 - 1 + BINS] += green_value;
			if (green_value > 0 && hist_min[1] == INT_MAX)
			{
				hist_min[1] = green_value;
			}

			int blue_value = hist_cum[j + i - 1 + BINS * 2];
			hist_cum[j + i * 2 - 1 + BINS * 2] += blue_value;
			if (blue_value > 0 && hist_min[2] == INT_MAX)
			{
				hist_min[2] = blue_value;
			}
		}
		__syncthreads();
	}

	if (threadIdx.x == 0)
	{
		hist_cum[BINS - 1] = 0;
		hist_cum[BINS * 2 - 1] = 0;
		hist_cum[BINS * 3 - 1] = 0;
	}

	__syncthreads();

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
		__syncthreads();
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

		GetHistogramGpu << <grid_size, block_size >> >(d_image, d_histogram, width, height, cpp);
		checkCudaErrors(
			cudaMemcpy(hist, d_histogram, HIST_CHANNELS * BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
#else
		GetHistogramCpu(image_in, hist, width, height, cpp);
#endif

		/*auto hist_cum_normal = static_cast<unsigned int*>(malloc(HIST_CHANNELS * BINS * sizeof(unsigned int)));
		memcpy(hist_cum_normal, hist, HIST_CHANNELS * BINS * sizeof(unsigned int));
		BasicCumHistCpu(hist_cum_normal);*/

		auto hist_cum = static_cast<unsigned int*>(malloc(HIST_CHANNELS * BINS * sizeof(unsigned int)));
		memcpy(hist_cum, hist, HIST_CHANNELS * BINS * sizeof(unsigned int));

		auto hist_mins = static_cast<unsigned int*>(malloc(HIST_CHANNELS * sizeof(unsigned int)));
		for (unsigned int i = 0; i < HIST_CHANNELS; i++)
		{
			hist_mins[i] = INT_MAX;
		}

#ifdef GPU
		BlellochScanHistGpu(hist_cum, hist_mins, false);
#else
		BlellochScanHistCpu(hist_cum, hist_mins, false);
#endif


		auto image_out = static_cast<unsigned char*>(malloc(width * height * 4 * sizeof(unsigned char)));

#ifdef GPU
		unsigned char* d_image_out;
		checkCudaErrors(cudaMalloc((void**)&d_image_out, height * width * 4 * sizeof(unsigned char)));
		unsigned int* d_hist_cum;
		unsigned int* d_hist_mins;
		checkCudaErrors(cudaMalloc((void**)&d_hist_cum, HIST_CHANNELS * BINS * sizeof(unsigned int)));
		checkCudaErrors(cudaMalloc((void**)&d_hist_mins, HIST_CHANNELS * sizeof(unsigned int)));
		checkCudaErrors(
			cudaMemcpy(d_hist_cum, hist_cum, HIST_CHANNELS * BINS * sizeof(unsigned int), cudaMemcpyHostToDevice));
		checkCudaErrors(
			cudaMemcpy(d_hist_mins, hist_mins, HIST_CHANNELS * sizeof(unsigned int), cudaMemcpyHostToDevice));

		EqualizeImageGpu << < grid_size, block_size >> >(height, width, d_image, d_image_out, d_hist_cum, d_hist_mins);
		checkCudaErrors(
			cudaMemcpy(image_out, d_image_out, height * width * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

		cudaFree(d_image);
		cudaFree(d_histogram);
#else
		EqulizeImageCpu(height, width, image_in, image_out, hist_cum, hist_mins);

#endif
		cudaEventRecord(stop);

		cudaEventSynchronize(stop);

		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("Time: %0.3f milliseconds \n", milliseconds);

		stbi_write_png("lena_result.png", width, height, 4, image_out, width * 4);
		free(hist_cum);
		free(image_out);
	}
	else
	{
		fprintf(stderr, "Error loading image %s!\n", image_file);
	}

	return 0;
}

void EqulizeImageCpu(int height, int width, unsigned char* image_in, unsigned char* image_out, unsigned int* hist_cum,
                     unsigned int* hist_mins)
{
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			const int index = (i * width + j) * 4;
			int r = image_in[index];
			int g = image_in[index + 1];
			int b = image_in[index + 2];

			image_out[index] = round(
				((static_cast<double>(hist_cum[r] - hist_mins[0])) / static_cast<double>((height * width - hist_mins[
					0]))) * (BINS - 1));
			image_out[index + 1] = round(
				((static_cast<double>(hist_cum[g + BINS] - hist_mins[1])) / static_cast<double>((height * width -
					hist_mins[1]))) * (BINS - 1));
			image_out[index + 2] = round(
				((static_cast<double>(hist_cum[b + BINS * 2] - hist_mins[2])) / static_cast<double>((height * width -
					hist_mins[2]))) * (BINS - 1));
			image_out[index + 3] = image_in[index + 3]; // Keep the A value
		}
	}
}

void BlellochScanHistGpu(unsigned int* hist_cum, unsigned int* hist_mins, bool print)
{
	unsigned int* d_hist_cum;
	unsigned int* d_hist_mins;
	checkCudaErrors(cudaMalloc((void**)&d_hist_cum, HIST_CHANNELS * BINS * sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_hist_mins, HIST_CHANNELS * sizeof(unsigned int)));
	checkCudaErrors(
		cudaMemcpy(d_hist_cum, hist_cum, HIST_CHANNELS * BINS * sizeof(unsigned int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_hist_mins, hist_mins, HIST_CHANNELS * sizeof(unsigned int), cudaMemcpyHostToDevice));

	dim3 block_size(BINS, 1, 1);
	dim3 grid_size(1, 1, 1);

	BlellochScanHistKernel << <grid_size, block_size >> >(d_hist_cum, d_hist_mins);

	checkCudaErrors(
		cudaMemcpy(hist_cum, d_hist_cum, HIST_CHANNELS * BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(hist_mins, d_hist_mins, HIST_CHANNELS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

	if (print)
	{
		PrintHistogram(hist_cum);
		printf("Log: min values: r%d g%d b%d\n", hist_mins[0], hist_mins[1], hist_mins[2]);
	}

	cudaFree(d_hist_cum);
	cudaFree(d_hist_mins);
}

void BlellochScanHistCpu(unsigned int* hist_cum, unsigned int* hist_min, bool print)
{
	for (int i = 1; i < BINS; i *= 2)
	{
		for (int j = 0; j < BINS; j += i * 2)
		{
			int red_value = hist_cum[j + i - 1];
			hist_cum[j + i * 2 - 1] += red_value;
			if (red_value > 0 && hist_min[0] == 99999)
			{
				hist_min[0] = red_value;
			}

			int green_value = hist_cum[j + i - 1 + BINS];
			hist_cum[j + i * 2 - 1 + BINS] += green_value;
			if (green_value > 0 && hist_min[1] == 99999)
			{
				hist_min[1] = green_value;
			}

			int blue_value = hist_cum[j + i - 1 + BINS * 2];
			hist_cum[j + i * 2 - 1 + BINS * 2] += blue_value;
			if (blue_value > 0 && hist_min[2] == 99999)
			{
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
	if (print)
	{
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
