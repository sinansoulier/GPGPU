#pragma once

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void filter_impl(uint8_t* buffer1, uint8_t* buffer2, int width, int height, int stride, int pixel_stride);

#ifdef __cplusplus
}
#endif
