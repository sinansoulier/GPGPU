#ifndef LAB_CUDA_H
#define LAB_CUDA_H

__global__ void compute_lab_image(std::byte* background, std::byte* current_frame, int stride, int width, int height);

#endif