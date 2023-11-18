#ifndef UTILS_H
#define UTILS_H


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

struct FrameStorage {
    uint8_t* average_background;  
    int frame_count = 0;
    int width, height, pixel_stride;
};

#endif