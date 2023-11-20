#include "filter_impl.h"

#include <cassert>
#include <chrono>
#include <thread>
#include <cstdio>
#include "implem_cpp/utils.hh"

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

__device__ float linearize(float channel) {
    if (channel <= 0.04045)
        return channel / 12.92;

    return pow((channel + 0.055) / 1.055, 2.4);
}

__device__ void rgb_to_xyz(rgb* rgb, xyz* xyz)
{
    float r = linearize(rgb->r);
    float g = linearize(rgb->g);
    float b = linearize(rgb->b);

    // Convert from sRGB to the CIE XYZ color space using the D65 illuminant.
    // The conversion matrix is based on the D65 illuminant (a standard form of daylight).
    xyz->x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    xyz->y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    xyz->z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;
}

__device__ void xyz_to_lab(xyz* xyz, lab* lab)
{
    // The reference white point for D65 illuminant in the XYZ color space.
    float ref_X =  0.95047; 
    float ref_Y =  1.00000;
    float ref_Z =  1.08883;

    // Check if the XYZ values are scaled between 0 to 255, and if so, normalize them to 0 to 1 range.
    if (xyz->x > 1 || xyz->y > 1 || xyz->z > 1){
        xyz->x = xyz->x / 255.0f;
        xyz->y = xyz->y / 255.0f;
        xyz->z = xyz->z / 255.0f;
    }

    // Normalize the XYZ values with the reference white point.
    float x = xyz->x / ref_X;
    float y = xyz->y / ref_Y;
    float z = xyz->z / ref_Z;

    // Convert XYZ to Lab. This involves a piecewise function for each coordinate.
    // The constants and equations come from the official Lab color space definition.
    x = (x > 0.008856) ? pow(x, 1.0/3.0) : (7.787 * x + 16.0/116.0);
    y = (y > 0.008856) ? pow(y, 1.0/3.0) : (7.787 * y + 16.0/116.0);
    z = (z > 0.008856) ? pow(z, 1.0/3.0) : (7.787 * z + 16.0/116.0);
    
    // Compute the L*, a*, and b* values from the transformed x, y, z coordinates.
    lab->l = (116.0 * y) - 16.0;
    lab->a = 500.0 * (x - y);
    lab->b = 200.0 * (y - z);
}

//ΔEab ​= sqrt(L2​−L1​)**2 + (a2​−a1​)**2 + (b2​−b1​)**2)
__device__ float compute_lab(rgb* pix_img_1, rgb* pix_img_2)
{
    xyz xyz1 = { 0, 0, 0 };
    xyz xyz2 = { 0, 0, 0};
    rgb_to_xyz(pix_img_1, &xyz1);
    rgb_to_xyz(pix_img_2, &xyz2);

    // lab lab1, lab2;
    lab lab1 = { 0, 0, 0 };
    lab lab2 = { 0, 0, 0 };
    xyz_to_lab(&xyz1, &lab1);
    xyz_to_lab(&xyz2, &lab2);
    
    float norm_coeff = 0.01;
    float luminance_diff = pow(lab2.l - lab1.l, 2) * norm_coeff;
    float a_diff = pow(lab2.a - lab1.a, 2) * norm_coeff;
    float b_diff = pow(lab2.b - lab1.b, 2) * norm_coeff;
    
    float lab = sqrt(luminance_diff + a_diff + b_diff);
    return lab;
}

__global__ void compute_lab_image(std::byte* background, std::byte* current_frame, int stride, int width, int height)
{
    int y = blockIdx.y * blockDim.y + threadIdx.y; 
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x >= width || y >= height)
        return; 

    rgb* lineptr1 = (rgb*) (background + y * stride);
    rgb* lineptr2 = (rgb*) (current_frame + y * stride);

    float lab = compute_lab(&lineptr1[x], &lineptr2[x]);

    lineptr2[x].r = lab;
    lineptr2[x].g = lab;
    lineptr2[x].b = lab;
}

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

__global__ void hysterisis_mark(std::byte* image, int low_threshold, int high_threshold, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb* pixel = (rgb*)(image + y * stride + x * sizeof(rgb));
    float current = pixel->r;

    if (current > high_threshold) {
        pixel->r = pixel->g = pixel->b = 255;
    } else if (current < low_threshold) {
        pixel->r = pixel->g = pixel->b = 0;
    } else {
        pixel->r = pixel->g = pixel->b = 128;
    }
}

__global__ void hysterisis_compute(std::byte* image, int low_threshold, int high_threshold, int width, int height, int stride)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    rgb* current_pixel = (rgb*)(image + y * stride + x * sizeof(rgb));
    if (current_pixel->r == 128) {
        for (int dx = -1; dx <= 1; dx++) {
            for (int dy = -1; dy <= 1; dy++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                    rgb* neighbor_pixel = (rgb*)(image + ny * stride + nx * sizeof(rgb));
                    if (neighbor_pixel->r == 255) {
                        current_pixel->r = current_pixel->g = current_pixel->b = 255;
                    }
                }
            }
        }

        current_pixel->r = current_pixel->g = current_pixel->b = 0;
    }
}

namespace
{
    static FrameStorage frame_storage = { 0 };
}

extern "C" {
    void filter_impl(uint8_t* src_buffer, int width, int height, int src_stride, int pixel_stride)
    {
        if (frame_storage.frame_count == 0)
        {
            frame_storage.frame_count = 1;
            frame_storage.width = width;
            frame_storage.height = height;
            frame_storage.pixel_stride = src_stride;

            uint8_t* background = new uint8_t[width * height * pixel_stride];
            memcpy(background, src_buffer, width * height * pixel_stride);

            frame_storage.average_background = background;
            frame_storage.frame_count++;
        }

        assert(sizeof(rgb) == pixel_stride);
        std::byte* dBuffer;
        std::byte* dBackground;
        size_t pitch;

        cudaError_t err;
        
        err = cudaMallocPitch(&dBuffer, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBuffer, pitch, src_buffer, src_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        err = cudaMallocPitch(&dBackground, &pitch, width * sizeof(rgb), height);
        CHECK_CUDA_ERROR(err);
        err = cudaMemcpy2D(dBackground, pitch, frame_storage.average_background, frame_storage.pixel_stride, width * sizeof(rgb), height, cudaMemcpyHostToDevice);
        CHECK_CUDA_ERROR(err);

        dim3 blockSize(16,16);
        dim3 gridSize((width + (blockSize.x - 1)) / blockSize.x, (height + (blockSize.y - 1)) / blockSize.y);

        compute_lab_image<<<gridSize, blockSize>>>(dBackground, dBuffer, pitch, width, height);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        opening<<<gridSize, blockSize>>>(dBuffer, 3, width, height, pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        int low_threshold = 4;
        int high_threshold = 30;

        hysterisis_mark<<<gridSize, blockSize>>>(dBuffer, low_threshold, high_threshold, pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        hysterisis_compute<<<gridSize, blockSize>>>(dBuffer, low_threshold, high_threshold, width, height, pitch);
        err = cudaDeviceSynchronize();
        CHECK_CUDA_ERROR(err);

        err = cudaMemcpy2D(src_buffer, src_stride, dBuffer, pitch, width * sizeof(rgb), height, cudaMemcpyDeviceToHost);
        CHECK_CUDA_ERROR(err);

        cudaFree(dBuffer);
        cudaFree(dBackground);

        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
