#include "lab_cuda.cuh"

#include <cmath>
#include <cstdio>
#include "../implem_cpp/utils.hh"

/// Wrapper for pow function to be used in device code
/// @param x base
/// @param y exponent
__device__ float devicePow(float x, float y)
{
    return pow(x, y);
}

/// Wrapper for sqrt function to be used in device code
/// @param x value to compute the square root of
__device__ float deviceSqrt(float x)
{
    return sqrt(x);
}

/// Linearization fuunction for sRGB
/// @param channel value to linearize
__device__ float linearize(float channel) {
    if (channel <= 0.04045)
        return channel / 12.92;

    return devicePow((channel + 0.055) / 1.055, 2.4);
}

/// Convert RGB to spatially uniform CIE XYZ color space
/// @param canals pointer to the RGB pixel
/// @param coordinates pointer to the XYZ pixel
__device__ void rgb_to_xyz(rgb* canals, xyz* coordinates)
{
    auto r = canals->r;
    auto g = canals->g;
    auto b = canals->b;

    float val = linearize((r + g + b) / 3.0f);

    coordinates->x = val * 0.4124564 + val * 0.3575761 + val * 0.1804375;
    coordinates->y = val * 0.2126729 + val * 0.7151522 + val * 0.0721750;
    coordinates->z = val * 0.0193339 + val * 0.1191920 + val * 0.9503041;
}

/// Convert XYZ to CIE LAB color space
/// @param coordinates pointer to the XYZ pixel
/// @param lab_coordinates pointer to the LAB pixel
__device__ void xyz_to_lab(xyz* coordinates, lab* lab_coordinates)
{
    // The reference white point for D65 illuminant in the XYZ color space.
    float ref_X =  0.95047; 
    float ref_Y =  1.00000;
    float ref_Z =  1.08883;

    // Check if the XYZ values are scaled between 0 to 255, and if so, normalize them to 0 to 1 range.
    if (coordinates->x > 1 || coordinates->y > 1 || coordinates->z > 1)
    {
        coordinates->x = coordinates->x / 255.0f;
        coordinates->y = coordinates->y / 255.0f;
        coordinates->z = coordinates->z / 255.0f;
    }

    // Normalize the XYZ values with the reference white point.
    float x = coordinates->x / ref_X;
    float y = coordinates->y / ref_Y;
    float z = coordinates->z / ref_Z;

    // Convert XYZ to Lab. This involves a piecewise function for each coordinate.
    // The constants and equations come from the official Lab color space definition.
    x = (x > 0.008856) ? devicePow(x, 1.0/3.0) : (7.787 * x + 16.0/116.0);
    y = (y > 0.008856) ? devicePow(y, 1.0/3.0) : (7.787 * y + 16.0/116.0);
    z = (z > 0.008856) ? devicePow(z, 1.0/3.0) : (7.787 * z + 16.0/116.0);
    
    // Compute the L*, a*, and b* values from the transformed x, y, z coordinates.
    lab_coordinates->l = (116.0 * y) - 16.0;
    lab_coordinates->a = 500.0 * (x - y);
    lab_coordinates->b = 200.0 * (y - z);
}

// ΔEab ​= sqrt((L2​−L1​)**2 + (a2​−a1​)**2 + (b2​−b1​)**2)
__device__ float compute_lab(rgb* pix_img_1, rgb* pix_img_2)
{
    __shared__ xyz xyz1;
    __shared__ xyz xyz2;
    __shared__ lab lab1;
    __shared__ lab lab2;

    rgb_to_xyz(pix_img_1, &xyz1);
    xyz_to_lab(&xyz1, &lab1);

    rgb_to_xyz(pix_img_2, &xyz2);
    xyz_to_lab(&xyz2, &lab2);

    float norm_coeff = 0.01;
    float luminance_diff = devicePow(lab2.l - lab1.l, 2);
    float a_diff = devicePow(lab2.a - lab1.a, 2);
    float b_diff = devicePow(lab2.b - lab1.b, 2);
    
    return deviceSqrt((luminance_diff + a_diff + b_diff) * norm_coeff);
}

#define TILE_SIZE 16

__global__ void compute_lab_image(std::byte* background, std::byte* current_frame, int stride, int width, int height)
{
    const int size = TILE_SIZE + 2;
    __shared__ rgb pix_img_1[size * size], pix_img_2[size * size];
    
    int bx = blockIdx.x * blockDim.x;
    int by = blockIdx.y * blockDim.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = bx + tx;
    int y = by + ty;

    int x_tile = x - 1;
    int y_tile = y - 1;

    if (x >= width || y >= height)
        return;
    
    __syncthreads();
    
    int index = (ty + 1) * TILE_SIZE + (tx + 1);
    pix_img_1[index] = *((rgb*) (background + y_tile * stride + x_tile * sizeof(rgb)));
    pix_img_2[index] = *((rgb*) (current_frame + y_tile * stride + x_tile * sizeof(rgb)));
    __syncthreads();

    auto lab = (uint8_t) compute_lab(&(pix_img_1[index]), &(pix_img_2[index]));
    __syncthreads();

    *((rgb*) (current_frame + y_tile * stride + x_tile * sizeof(rgb))) = {lab};
}