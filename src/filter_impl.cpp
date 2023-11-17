#include "filter_impl.h"

#include <chrono>
#include <thread>
#include "logo.h"
#include <cmath>
#include <cstring>

#include <iostream>

struct rgb {
    uint8_t r, g, b;
};

struct lab {
    float l, a, b;
};

struct xyz {
    float x, y, z;
};

typedef struct {
  uint8_t* first_frame;
  bool is_saved;
} FrameStorage;

// Variable globale pour la structure
static FrameStorage frame_storage = {0};

extern "C" {
    float linearize(float channel) {
        if (channel <= 0.04045)
            return channel / 12.92;

        return pow((channel + 0.055) / 1.055, 2.4);
    }

    void rgb_to_xyz(rgb* rgb, xyz* xyz)
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

    void xyz_to_lab(xyz* xyz, lab* lab)
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
    float compute_lab(rgb* pix_img_1, rgb* pix_img_2)
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
        
        float luminance_diff = pow(lab2.l - lab1.l, 2);
        float a_diff = pow(lab2.a - lab1.a, 2);
        float b_diff = pow(lab2.b - lab1.b, 2);
        
        float lab = sqrt(luminance_diff + a_diff + b_diff);
        return lab;
    }

    void compute_lab_image(uint8_t* buffer , int width, int height, int stride, int pixel_stride)
    {
        // Allocate new image
        // uint8_t* lab_image = new uint8_t[width * height];
        if (buffer == nullptr)
            return;
        
        bool is_equal = true;
        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr1 = (rgb*) (frame_storage.first_frame + y * stride);
            rgb* lineptr2 = (rgb*) (buffer + y * stride);
            for (int x = 0; x < width; ++x)
            {
                float lab = compute_lab(&lineptr1[x], &lineptr2[x]);
                is_equal = lineptr1[x].r != lineptr2[x].r || lineptr1[x].g != lineptr2[x].g || lineptr1[x].b != lineptr2[x].b;

                lineptr2[x].r = lab;
                lineptr2[x].g = lab;
                lineptr2[x].b = lab;
            }
        }
    }

    // Get pixel_value of a pixel after erosion
    float erode(uint8_t* image, int x, int y, int kernel_size, int width, int height)
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
    void erosion(uint8_t* image, int kernel_size, int width, int height)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; j++)
                ((rgb*) (image + i))[j].r = erode(image, i, j, kernel_size, width, height);
        }
    }

    // Get pixel_value of a pixel after dilatation
    float dilate(uint8_t* image, int x, int y, int kernel_size, int width, int height)
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
    
    void dilatation(uint8_t* image, int kernel_size, int width, int height)
    {
        for (int i = 0; i < height; i++)
        {
            for (int j = 0; j < width; ++j)
                ((rgb*) (image + i))[j].r = dilate(image, i, j, kernel_size, width, height);
        }
    }

    void opening(uint8_t* image, int kernel_size, int width, int height)
    {
        erosion(image, kernel_size, width, height);
        dilatation(image, kernel_size, width, height);
    }

    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        // On sauvegarde la première frame
        if (!frame_storage.is_saved)
        {
            uint8_t* pixels_copy = new uint8_t[width * height * pixel_stride];
            memcpy(pixels_copy, buffer, width * height * pixel_stride);
            frame_storage.first_frame = pixels_copy;
            frame_storage.is_saved = true;
        }
        else
        {
            compute_lab_image(buffer, width, height, stride, pixel_stride);
            opening(buffer, 3, width, height);
        }
        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
