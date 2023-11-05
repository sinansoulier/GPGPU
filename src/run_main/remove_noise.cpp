#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>


// Get pixel_value of a pixel after erosion
float erode(cv::Mat image, int x, int y, int kernel_size){
    int radius = kernel_size / 2;
    float current_value = image.at<float>(x, y);
    

    for (int i = -radius; i <= radius; i++){
        for (int j = -radius; j <= radius; j++){
            int x_pos = x + i;
            int y_pos = y + j;

            if (x_pos >= 0 && x_pos < image.rows &&
                y_pos >= 0 && y_pos < image.cols){
                float value = image.at<float>(x_pos, y_pos);
                if (value < current_value)
                    current_value = value;
            }
        }
    }
   
    return current_value;
}


// Apply erosion to an image
cv::Mat erosion(cv::Mat image, int kernel_size){
    cv::Mat eroded_image = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);

    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            eroded_image.at<float>(i, j) = erode(image, i, j, kernel_size);
        }
    }

    return eroded_image;
}


// Get pixel_value of a pixel after dilatation
float dilate(cv::Mat image, int x, int y, int kernel_size){
    int radius = kernel_size / 2;
    float current_value = image.at<float>(x, y);

    for (int i = -radius; i <= radius; i++){
        for (int j = -radius; j <= radius; j++){
            int x_pos = x + i;
            int y_pos = y + j;

            if (x_pos >= 0 && x_pos < image.rows &&
                y_pos >= 0 && y_pos < image.cols){
                float value = image.at<float>(x_pos, y_pos);
                if (value > current_value)
                    current_value = value;
            }
        }
    }

    return current_value;
}


// Apply dilatation to an image
cv::Mat dilatation(cv::Mat image, int kernel_size){
    cv::Mat dilated_image = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);

    for (int i = 0; i < image.rows; i++){
        for (int j = 0; j < image.cols; j++){
            dilated_image.at<float>(i, j) = dilate(image, i, j, kernel_size);
        }
    }

    return dilated_image;
}

// Apply opening to an image
cv::Mat opening(cv::Mat image, int kernel_size){
    cv::Mat eroded_image = erosion(image, kernel_size);
    cv::Mat opened_image = dilatation(eroded_image, kernel_size);

    return opened_image;
}