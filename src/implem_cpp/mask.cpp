#include "utils.hh"


void mask(uint8_t* binaryImage, uint8_t* originalImage, int width, int height, float alpha) {
    const rgb redTint = {255, 0, 0};

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            rgb* binaryPixel = (rgb*)(binaryImage + i * width * sizeof(rgb) + j * sizeof(rgb));
            rgb* originalPixel = (rgb*)(originalImage + i * width * sizeof(rgb) + j * sizeof(rgb));

            if (binaryPixel->r == 255) {  
                originalPixel->r = static_cast<uint8_t>(alpha * redTint.r + (1 - alpha) * originalPixel->r);
                originalPixel->g = static_cast<uint8_t>(alpha * redTint.g + (1 - alpha) * originalPixel->g);
                originalPixel->b = static_cast<uint8_t>(alpha * redTint.b + (1 - alpha) * originalPixel->b);
            }
        }
    }
}