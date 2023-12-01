#ifndef HYSTERISIS_CUDA_H
#define HYSTERISIS_CUDA_H

__global__ void hysterisis_mark(std::byte* image, int low_threshold, int high_threshold, int stride);
__global__ void hysterisis_compute(std::byte* image, int low_threshold, int high_threshold, int width, int height, int stride);

#endif