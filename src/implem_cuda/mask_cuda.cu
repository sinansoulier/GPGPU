#include "../implem_cpp/utils.hh"
#include "mask_cuda.cuh"

#include <cstdio>

__global__ void mask(std::byte* binaryImage, std::byte* originalImage, int width, int height, int stride, float alpha)
{
    const rgb redTint = { 255, 0, 0 };

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb* binaryPixel = (rgb*)(binaryImage + y * stride + x * sizeof(rgb));
    rgb* originalPixel = (rgb*)(originalImage + y * stride + x * sizeof(rgb));

    if (binaryPixel->r == 255) {
        originalPixel->r = static_cast<uint8_t>(alpha * originalPixel->r + (1 - alpha) * redTint.r);
        originalPixel->g = static_cast<uint8_t>(alpha * originalPixel->g + (1 - alpha) * redTint.g);
        originalPixel->b = static_cast<uint8_t>(alpha * originalPixel->b + (1 - alpha) * redTint.b);
    }
}