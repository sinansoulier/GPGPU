#ifndef MASK_CUDA_H
#define MASK_CUDA_H

__global__ void mask(std::byte* binaryImage, std::byte* originalImage, int width, int height, int stride, float alpha);

#endif