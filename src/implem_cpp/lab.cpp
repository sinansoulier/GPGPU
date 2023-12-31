#include "utils.hh"

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
    
    float norm_coeff = 0.01;
    float luminance_diff = pow(lab2.l - lab1.l, 2) * norm_coeff;
    float a_diff = pow(lab2.a - lab1.a, 2) * norm_coeff;
    float b_diff = pow(lab2.b - lab1.b, 2) * norm_coeff;
    
    float lab = sqrt(luminance_diff + a_diff + b_diff);
    return lab;
}

void compute_lab_image(uint8_t* current_frame, uint8_t* background, int width, int height, int stride, int pixel_stride)
{
    if (current_frame == nullptr)
        return;
    
    for (int y = 0; y < height; ++y)
    {
        rgb* lineptr1 = (rgb*) (background + y * stride);
        rgb* lineptr2 = (rgb*) (current_frame + y * stride);
        for (int x = 0; x < width; ++x)
        {
            float lab = compute_lab(&lineptr1[x], &lineptr2[x]);

            lineptr2[x].r = lab;
            lineptr2[x].g = lab;
            lineptr2[x].b = lab;
        }
    }
}

