#include "utils.hh"


void update_background(uint8_t* current_frame, FrameStorage& frame_storage) {
    if (frame_storage.frame_count == 0 || frame_storage.frame_count == 1)
        return;

    int width = frame_storage.width; 

    // Compute average background
    for (int i = 0; i < frame_storage.height; i++) {
        for (int j = 0; j < frame_storage.width; j++) {
            rgb* current_pixel = (rgb*)(frame_storage.average_background + i * width * sizeof(rgb) + j * sizeof(rgb));
            rgb* frame_pixel = (rgb*)(current_frame + i * width * sizeof(rgb) + j * sizeof(rgb));

            int n = frame_storage.frame_count - 1;

            current_pixel->r = (current_pixel->r * n + frame_pixel->r) / (n+1);
            current_pixel->g = (current_pixel->g * n + frame_pixel->g) / (n+1);
            current_pixel->b = (current_pixel->b * n + frame_pixel->b) / (n+1);
        }
    }
}