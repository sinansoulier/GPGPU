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

extern __shared__ rgb sharedMem[];

__global__ void hysterisis_compute(std::byte* image, int low_threshold, int high_threshold, int width, int height, int stride)
{
    const int TILE_WIDTH = blockDim.x + 2; 
    int index = (threadIdx.y + 1) * TILE_WIDTH + (threadIdx.x + 1);
    
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int x = bx + tx;
    int y = by + ty;

    int loaded_x = x - 1;
    int loaded_y = y - 1;

    if (loaded_x >= 0 && loaded_x < width && loaded_y >= 0 && loaded_y < height) {
        sharedMem[index] = *((rgb*)(image + loaded_y * stride + loaded_x * sizeof(rgb)));
    }

    __syncthreads();

    if (x < width && y < height && tx > 0 && tx < blockDim.x - 1 && ty > 0 && ty < blockDim.y - 1) {
        rgb current_pixel = sharedMem[index];
        if (current_pixel.r == 128) {
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    int neighbor_index = index + dy * TILE_WIDTH + dx; 
                    rgb neighbor_pixel = sharedMem[neighbor_index];
                    if (neighbor_pixel.r == 255) {
                        current_pixel.r = current_pixel.g = current_pixel.b = 255;
                        break;
                    }
                }
            }
            sharedMem[index] = current_pixel;
        }
    }

    __syncthreads();

    if (x < width && y < height) {
        *((rgb*)(image + y * stride + x * sizeof(rgb))) = sharedMem[index];
    }
}
