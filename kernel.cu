#define _CRT_SECURE_NO_WARNINGS
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;
using namespace std::chrono;

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << " " << line << endl;
        if (abort) exit(code);
    }
}

#define TILE_WIDTH 32

__global__ void resizeKernel(
    unsigned char* input, unsigned char* output,
    int width, int height, int new_width, int new_height, int channels,
    float x_ratio, float y_ratio)
{
    int x_out = blockIdx.x * blockDim.x + threadIdx.x;
    int y_out = blockIdx.y * blockDim.y + threadIdx.y;

    if (x_out >= new_width || y_out >= new_height) return;

    int px = min(int(x_out * x_ratio), width - 1);
    int py = min(int(y_out * y_ratio), height - 1);

    int out_idx = (y_out * new_width + x_out) * channels;
    int in_idx = (py * width + px) * channels;

    for (int c = 0; c < channels; c++) {
        output[out_idx + c] = input[in_idx + c];
    }
}

int main() {
    int width, height, channels;

    auto start = high_resolution_clock::now();

    unsigned char* img = stbi_load("chemistry.jpg", &width, &height, &channels, 0);
    if (!img) {
        cerr << "Eroare la incarcarea imaginii!\n";
        return -1;
    }

    int new_width = 5000;
    int new_height = 5000;

    size_t input_size = static_cast<size_t>(width) * height * channels;
    size_t output_size = static_cast<size_t>(new_width) * new_height * channels;

    unsigned char* d_input = nullptr;
    unsigned char* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input, input_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    CUDA_CHECK(cudaMemcpy(d_input, img, input_size, cudaMemcpyHostToDevice));

    float x_ratio = float(width) / new_width;
    float y_ratio = float(height) / new_height;

    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((new_width + TILE_WIDTH - 1) / TILE_WIDTH, (new_height + TILE_WIDTH - 1) / TILE_WIDTH);

    resizeKernel << <gridSize, blockSize >> > (
        d_input, d_output,
        width, height, new_width, new_height, channels, x_ratio, y_ratio);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned char* resized_img = new unsigned char[output_size];
    CUDA_CHECK(cudaMemcpy(resized_img, d_output, output_size, cudaMemcpyDeviceToHost));

    stbi_write_jpg("output5.jpg", new_width, new_height, channels, resized_img, 100);

    stbi_image_free(img);
    delete[] resized_img;

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    cout << "Imaginea a fost redimensionata si salvata!\n";
    cout << "Timpul de redimensionare (cu CUDA): " << elapsed.count() << " secunde\n";

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    cout << "Memorie libera: " << free_mem / (1024 * 1024) << " MB din " << total_mem / (1024 * 1024) << " MB\n";

    return 0;
}
