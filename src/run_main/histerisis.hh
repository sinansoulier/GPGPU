#ifndef HISTERISIS_H
#define HISTERISIS_H
#include <opencv2/opencv.hpp>

cv::Mat histerisis(cv::Mat image, int low_pass, int hight_pass);


#endif