#include "filter_impl.h"

#include <chrono>
#include <thread>
#include "logo.h"
#include <cmath>

struct rgb {
    uint8_t r, g, b;
};

struct lab {
    float l, a, b;
};

struct xyz {
    float x, y, z;
};

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

    void xyz_to_lab(xyz* xgb, lab* lab)
    {
        // The reference white point for D65 illuminant in the XYZ color space.
        float ref_X =  0.95047; 
        float ref_Y =  1.00000;
        float ref_Z =  1.08883;

        // Check if the XYZ values are scaled between 0 to 255, and if so, normalize them to 0 to 1 range.
        if (xgb->x > 1 || xgb->y > 1 || xgb->z > 1){
            xgb->x = xgb->x / 255.0f;
            xgb->y = xgb->y / 255.0f;
            xgb->z = xgb->z / 255.0f;
        }

        // Normalize the XYZ values with the reference white point.
        float x = xgb->x / ref_X;
        float y = xgb->y / ref_Y;
        float z = xgb->z / ref_Z;

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
        xyz xyz1, xyz2;
        rgb_to_xyz(pix_img_1, &xyz1);
        rgb_to_xyz(pix_img_2, &xyz2);

        lab lab1, lab2;
        xyz_to_lab(&xyz1, &lab1);
        xyz_to_lab(&xyz2, &lab2);
        
        float luminance_diff = pow(lab2.l - lab1.l, 2);
        float a_diff = pow(lab2.a - lab1.a, 2);
        float b_diff = pow(lab2.b - lab1.b, 2);
        
        float lab = sqrt(luminance_diff + a_diff + b_diff);
        return lab;
    }

    void compute_lab_image(uint8_t* buffer1, uint8_t* buffer2 , int width, int height, int stride, int pixel_stride)
    {
        // Allocate new image
        // uint8_t* lab_image = new uint8_t[width * height];
        if (buffer1 == nullptr)
            return;

        for (int y = 0; y < height; ++y)
        {
            rgb* lineptr1 = (rgb*) (buffer1 + y * stride);
            rgb* lineptr2 = (rgb*) (buffer2 + y * stride);
            for (int x = 0; x < width; ++x)
            {
                float lab = compute_lab(&lineptr1[x], &lineptr2[x]);

                lineptr2[x].r = lab;
                lineptr2[x].g = lab;
                lineptr2[x].b = lab;
            }
        }
    }

    void filter_impl(uint8_t* buffer1, uint8_t* buffer2, int width, int height, int stride, int pixel_stride)
    {
        compute_lab_image(buffer1, buffer2, width, height, stride, pixel_stride);

        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
