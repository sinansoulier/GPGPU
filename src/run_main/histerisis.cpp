#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>



cv::Mat compute_gradient(cv::Mat image){
    cv::Mat image_grad = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);
    
    for (int i = 1; i < image.rows - 1; i++) {
        for (int j = 1; j < image.cols - 1; j++) {
            float dx = image.at<float>(i + 1, j) - image.at<float>(i - 1, j);
            float dy = image.at<float>(i, j + 1) - image.at<float>(i, j - 1);
            image_grad.at<float>(i, j) = sqrt(dx * dx + dy * dy);
        }
    }

    return image_grad;
}


cv::Mat histerisis(const cv::Mat image, int low_threshold, int high_threshold) {
    cv::Mat gradient_magnitude = compute_gradient(image);
    // cv::Mat gradient_magnitude = image;
    cv::Mat edges = cv::Mat::zeros(image.rows, image.cols, CV_32FC1);

    for (int i = 0; i < gradient_magnitude.rows; i++) {
        for (int j = 0; j < gradient_magnitude.cols; j++) {
            float gradient = gradient_magnitude.at<float>(i, j);

            if (gradient > high_threshold) {
                edges.at<float>(i, j) = 255;
            } else if (gradient < low_threshold) {
                edges.at<float>(i, j) = 128;
            } else{
                edges.at<float>(i, j) = 0;
            }   
        }
    }

    for (int iter = 0; iter < 100; iter++){
        for (int i = 0; i < edges.rows; i++) {
            for (int j = 0; j < edges.cols; j++) {
                
                if (edges.at<float>(i, j) == 128) {
                    for (int dx = -1; dx <= 1; dx++) {
                        for (int dy = -1; dy <= 1; dy++) {
                            int nx = i + dx;
                            int ny = j + dy;

                            if (nx >= 0 && nx < edges.rows && ny >= 0 && ny < edges.cols) {
                                if (edges.at<float>(nx, ny) == 255) {
                                    edges.at<float>(i, j) = 255;
                                }
                            }
                        }
                    }

                    if (edges.at<float>(i, j) == 128) {
                        edges.at<float>(i, j) = 0;
                    }
                }
            }
        }
    }

    return edges;
}