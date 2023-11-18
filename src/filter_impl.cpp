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

static FrameStorage frame_storage = {};

extern "C" {
    void filter_impl(uint8_t* buffer, int width, int height, int stride, int pixel_stride)
    {
        if (frame_storage.frame_count == 0) {
            frame_storage.width = width;
            frame_storage.height = height;
            frame_storage.pixel_stride = pixel_stride;
            uint8_t* background = new uint8_t[width * height * pixel_stride];
            memcpy(background, buffer, width * height * pixel_stride);
            frame_storage.average_background = background;
            frame_storage.frame_count++;
        }
        
        // Updated Background
        uint8_t* current_background = new uint8_t[width * height * pixel_stride];
        memcpy(current_background, buffer, width * height * pixel_stride);
        update_background(current_background, frame_storage);
        frame_storage.frame_count++;


        // Filters
        uint8_t* current_frame = new uint8_t[width * height * pixel_stride];
        memcpy(current_frame, buffer, width * height * pixel_stride);
        compute_lab_image(current_frame, frame_storage.average_background,  width, height, stride, pixel_stride);
        opening(current_frame, 3, width, height);
        histerisis(current_frame, 4, 30, width, height);
        mask(current_frame, buffer, width, height, 0.7);


        // You can fake a long-time process with sleep
        {
            using namespace std::chrono_literals;
            //std::this_thread::sleep_for(100ms);
        }
    }   
}
