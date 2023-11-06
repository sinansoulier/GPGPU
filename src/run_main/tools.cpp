#include <opencv2/opencv.hpp>
#include <iostream>


cv::Mat load_image(std::string path, int type){
    cv::Mat image = cv::imread(path, type);
    if (image.empty()){
        std::cout << "Could not read the image: " << path << std::endl;
        exit(1);
    }

    return image;
}


bool save_image(const std::string& path, const cv::Mat& image) {
    if (cv::imwrite(path, image)) {
        std::cout << "Saved at " << path << std::endl;
        return true;
    } else {
        std::cerr << "Failed to save at " << path << std::endl;
        return false;
    }
}