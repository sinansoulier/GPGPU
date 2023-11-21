#include "background_update_cuda.cuh"
#include "../implem_cpp/utils.hh"

#include <cstdio>

__global__ void update_background(std::byte* current_frame, std::byte* average_background, int stride, int n)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb* current_pixel = (rgb*)(average_background + y * stride + x * sizeof(rgb));
    rgb* frame_pixel = (rgb*)(current_frame + y * stride + x * sizeof(rgb));

    current_pixel->r = (current_pixel->r * n + frame_pixel->r) / (n + 1);
    current_pixel->g = (current_pixel->g * n + frame_pixel->g) / (n + 1);
    current_pixel->b = (current_pixel->b * n + frame_pixel->b) / (n + 1);
}