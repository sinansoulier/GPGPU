#ifndef TOOLS_H
#define TOOLS_H
#include <opencv2/opencv.hpp>

cv::Mat load_image(std::string path);
bool save_image(const std::string& path, const cv::Mat& image);

#endif