#include "filter_impl.h"

#include <chrono>
#include <thread>
#include "logo.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <iostream>

#include "implem_cpp/utils.hh" 
#include "implem_cpp/lab.hh"
#include "implem_cpp/hysteresis.hh"
#include "implem_cpp/mask.hh"
#include "implem_cpp/remove_noise.hh"
#include "implem_cpp/background_update.hh"

static FrameStorage frame_storage = {0};
static int count = 0;

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        if (frame_storage.frame_count == 0) {
            frame_storage.width = width;
            frame_storage.height = height;
            frame_storage.pixel_stride = pixel_stride;

            uint8_t* background = new uint8_t[width * height * pixel_stride];
            memcpy(background, buffer, width * height * pixel_stride);

            frame_storage.image_buffer = new uint8_t[width * height * pixel_stride];
            
            frame_storage.average_background = background;
            frame_storage.frame_count++;
        }
        
        if (frame_storage.frame_count % 30 == 0)
        {
            // Updated Background
            update_background(buffer, frame_storage);
        }

        frame_storage.frame_count++;
        // Filters
        memcpy(frame_storage.image_buffer, buffer, width * height * pixel_stride);
        compute_lab_image(frame_storage.image_buffer, frame_storage.average_background,  width, height, stride, pixel_stride);
        opening(frame_storage.image_buffer, 3, width, height);
        histerisis(frame_storage.image_buffer, 4, 30, width, height);
        mask(frame_storage.image_buffer, buffer, width, height, 0.5);

        // memcpy(buffer, frame_storage.image_buffer, width * height * pixel_stride);

        // You can fake a long-time process with sleep
        // {
            // using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        // }
    }   
}
