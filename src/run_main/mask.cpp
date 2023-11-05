#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>


cv::Mat Mask(cv::Mat image, cv::Mat mask) {
    cv::Mat image_masked = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);

    if (image.empty() || mask.empty() || image.channels() != 3) {
        throw std::runtime_error("Invalid image or mask");
    }

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);

            float mask_value = mask.at<float>(i, j);
            if (mask_value == 255)
                mask_value = 1;
            else
                mask_value = 0;
            
            pixel[2] += 0.5 * pixel[2] * mask_value;
            if (pixel[2] > 255)
                pixel[2] = 255;

            image_masked.at<cv::Vec3b>(i, j) = pixel;
        }
    }

    return image_masked;
}