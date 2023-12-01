#include "../implem_cpp/utils.hh"
#include "mask_cuda.cuh"

#include <cstdio>

#define TILE_SIZE 16

__global__ void mask(std::byte* binaryImage, std::byte* originalImage, int width, int height, int stride, float alpha)
{
    const int size = TILE_SIZE + 1;
    __shared__ rgb binaryPixelLine[size * size];
    __shared__ rgb originalPixelLine[size * size];

    const rgb redTint = { 255, 0, 0 };

    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = bx + tx;
    int y = by + ty;

    int x_tile = x - 1;
    int y_tile = y - 1;

    if (x >= width || y >= height)
        return;

    __syncthreads();

    int index = (ty + 1) * TILE_SIZE + (tx + 1);
    binaryPixelLine[index] = *((rgb*) (binaryImage + y_tile * stride + x_tile * sizeof(rgb)));
    originalPixelLine[index] = *((rgb*) (originalImage + y_tile * stride + x_tile * sizeof(rgb)));

    __syncthreads();

    if (binaryPixelLine[index].r == 255) {
        originalPixelLine[index].r = (1 - alpha) * originalPixelLine[index].r + alpha * redTint.r;
        originalPixelLine[index].g = (1 - alpha) * originalPixelLine[index].g + alpha * redTint.g;
        originalPixelLine[index].b = (1 - alpha) * originalPixelLine[index].b + alpha * redTint.b;
    }

    __syncthreads();

    *((rgb*) (originalImage + y_tile * stride + x_tile * sizeof(rgb))) = originalPixelLine[index];
}