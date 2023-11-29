#include <cmath>
#include <cstring>
#include <iostream>

#include "utils.hh"

void histerisis(uint8_t* image, int low_threshold, int high_threshold, int width, int height) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                rgb* pixel = (rgb*)(image + i * width * sizeof(rgb) + j * sizeof(rgb));
                float current = pixel->r;  

                if (current > high_threshold) {
                    pixel->r = pixel->g = pixel->b = 255; 
                } else if (current < low_threshold) {
                    pixel->r = pixel->g = pixel->b = 0; 
                } else {
                    pixel->r = pixel->g = pixel->b = 128; 
                }
            }
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                rgb* current_pixel = (rgb*)(image + i * width * sizeof(rgb) + j * sizeof(rgb));
                if (current_pixel->r == 128) {
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            int nx = i + dx;
                            int ny = j + dy;

                            if (nx >= 0 && nx < height && ny >= 0 && ny < width) {
                                rgb* neighbor_pixel = (rgb*)(image + nx * width * sizeof(rgb) + ny * sizeof(rgb));
                                if (neighbor_pixel->r == 255) {
                                    current_pixel->r = current_pixel->g = current_pixel->b = 255;
                                }
                            }
                        }
                    }

                    current_pixel->r = current_pixel->g = current_pixel->b = 0;
                }
            }
        }
    }