#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include <iostream>
#include "implem_cpp/utils.hh"
#include "implem_cuda/background_update_cuda.cuh"
#include "implem_cuda/lab_cuda.cuh"
#include "implem_cuda/remove_noise_cuda.cuh"
#include "implem_cuda/hysterisis_cuda.cuh"
#include "implem_cuda/mask_cuda.cuh"

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "CUDA Runtime Error at: %s: %d\n", file, line);
        std::fprintf(stderr, "%s %s\n", cudaGetErrorString(err), func);
        // We don't exit when we encounter CUDA errors in this example.
        std::exit(EXIT_FAILURE);
    }
}

namespace
{
    class CudaFrameStorage
    {
    public:
        std::byte* average_background;
        int width;
        int height;
        int pixel_stride;
        int frame_count;
        ~CudaFrameStorage()
        {
            cudaFree(average_background);
        }
    };
    static CudaFrameStorage frame_storage = { 0 };
}

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        assert(sizeof(rgb) == pixel_stride);
        std::byte* dBuffer;
        std::byte* dMask;
        size_t pitch;
        cudaError_t err;

        if (frame_storage.frame_count == 0)
        {
            frame_storage.frame_count = 1;
            frame_storage.width = width;
            frame_storage.height = height;
            frame_storage.pixel_stride = src_stride;

            std::byte* background;
            err = cudaMallocPitch(&background, &pitch, width * sizeof(rgb), height);
            CHECK_CUDA_ERROR(err);
            err = cudaMemcpy2D(background, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
            CHECK_CUDA_ERROR(err);

            frame_storage.average_background = background;
            frame_storage.frame_count++;
        }
        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        err = cudaMallocPitch(&dMask, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dMask, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        if (frame_storage.frame_count % 30 == 0)
        {
            update_background<<<gridSize, blockSize>>>(dMask, frame_storage.average_background, pitch, frame_storage.frame_count - 1);
            err = cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(err);
        }

        frame_storage.frame_count++;

        int sharedMemSize = (blockSize.x + 2) * (blockSize.y + 2) * sizeof(rgb);

        compute_lab_image<<<gridSize, blockSize>>>(frame_storage.average_background, dMask, pitch, width, height);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        opening<<<gridSize, blockSize, sharedMemSize>>>(dMask, 3, width, height, pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        int low_threshold = 4;
        int high_threshold = 30;

        hysterisis_mark<<<gridSize, blockSize>>>(dMask, low_threshold, high_threshold, pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        hysterisis_compute<<<gridSize, blockSize, sharedMemSize>>>(dMask, low_threshold, high_threshold, width, height, pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        mask<<<gridSize, blockSize>>>(dMask, dBuffer, width, height, pitch, 0.5);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(dMask);

        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
