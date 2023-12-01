#ifndef BACKGROUND_UPDATE_CUDA_H
#define BACKGROUND_UPDATE_CUDA_H

__global__ void update_background(std::byte* current_frame, std::byte* average_background, int stride, int n);

#endif