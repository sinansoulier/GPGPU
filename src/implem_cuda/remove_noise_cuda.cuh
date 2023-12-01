#ifndef REMOVE_NOISE_CUDA_H
#define REMOVE_NOISE_CUDA_H

__global__ void opening(std::byte* image, int kernel_size, int width, int height, int stride);

#endif