#include "../implem_cpp/utils.hh"
#include "remove_noise_cuda.cuh"

#include <cstdio>

// Get pixel_value of a pixel after erosion
__device__ float erode(std::byte* image, int x, int y, int kernel_size, int width, int height)
{
    int radius = kernel_size / 2;
    float current_value = ((rgb*) (image + y))[x].r;
    
    for (int i = -radius; i <= radius; i++){
        for (int j = -radius; j <= radius; j++){
            int x_pos = x + i;
            int y_pos = y + j;

            if (x_pos >= 0 && x_pos < height &&
                y_pos >= 0 && y_pos < width)
                {
                float value = ((rgb*) (image + x_pos))[y_pos].r;
                if (value < current_value)
                    current_value = value;
            }
        }
    }

    return current_value;
}

// Apply erosion to an image
__device__ void erosion(std::byte* image, int kernel_size, int width, int height, int x, int y, int stride)
{
    ((rgb*) (image + y))[x].r = erode(image, x, y, kernel_size, width, height);
}

// Get pixel_value of a pixel after dilatation
__device__ float dilate(std::byte* image, int x, int y, int kernel_size, int width, int height)
{
    int radius = kernel_size / 2;
    float current_value = ((rgb*) (image + y))[x].r;

    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            int x_pos = x + i;
            int y_pos = y + j;

            if (x_pos >= 0 && x_pos < height && y_pos >= 0 && y_pos < width)
            {
                float value = ((rgb*) (image + x_pos))[y_pos].r;
                if (value > current_value)
                    current_value = value;
            }
        }
    }

    return current_value;
}

__device__ void dilatation(std::byte* image, int kernel_size, int width, int height, int x, int y, int stride)
{
    ((rgb*) (image + y))[x].r = dilate(image, x, y, kernel_size, width, height);
}

__global__ void opening(std::byte* image, int kernel_size, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    erosion(image, kernel_size, width, height, x, y, stride);
    dilatation(image, kernel_size, width, height, x, y, stride);
}