#include "utils.hh"


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