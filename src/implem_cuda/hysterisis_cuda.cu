#include "hysterisis_cuda.cuh"
#include "../implem_cpp/utils.hh"

#include <cstdio>

__global__ void hysterisis_mark(std::byte* image, int low_threshold, int high_threshold, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb* pixel = (rgb*)(image + y * stride + x * sizeof(rgb));
    float current = pixel->r;

    if (current > high_threshold) {
        pixel->r = pixel->g = pixel->b = 255;
    } else if (current < low_threshold) {
        pixel->r = pixel->g = pixel->b = 0;
    } else {
        pixel->r = pixel->g = pixel->b = 128;
    }
}

__global__ void hysterisis_compute(std::byte* image, int low_threshold, int high_threshold, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb* current_pixel = (rgb*)(image + y * stride + x * sizeof(rgb));
    if (current_pixel->r == 128) {
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                    rgb* neighbor_pixel = (rgb*)(image + ny * stride + nx * sizeof(rgb));
                    if (neighbor_pixel->r == 255) {
                        current_pixel->r = current_pixel->g = current_pixel->b = 255;
                    }
                }
            }
        }

        current_pixel->r = current_pixel->g = current_pixel->b = 0;
    }
}
