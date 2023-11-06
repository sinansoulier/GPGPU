#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>


cv::Mat Mask(cv::Mat image, cv::Mat mask) {
    cv::Mat image_masked = cv::Mat::zeros(image.rows, image.cols, CV_32FC3);
    std::cout << image.rows << " " << image.cols << std::endl;
    std::cout << image.at<cv::Vec3f>(0, 0) << std::endl;
    std::cout << mask.rows << " " << mask.cols << std::endl;
    std::cout << image.channels() << std::endl;


    if (image.empty() || mask.empty() || image.channels() != 3) {
        throw std::runtime_error("Invalid image or mask");
    }

    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            cv::Vec3f pixel = image.at<cv::Vec3f>(i, j);

            int mask_value = mask.at<int>(i, j);
            if (mask_value == 255)
                mask_value = 0;
            else
                mask_value = 1;
            
            pixel[2] += 0.5 * pixel[2] * mask_value;
            if (pixel[2] > 255)
                pixel[2] = 255;

            image_masked.at<cv::Vec3f>(i, j) = cv::Vec3f(pixel[0], pixel[1], pixel[2]);
        }
    }

    return image_masked;
}