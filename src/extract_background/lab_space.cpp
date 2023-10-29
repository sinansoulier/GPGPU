#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>


cv::Mat load_image(std::string path){
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()){
        std::cout << "Could not read the image: " << path << std::endl;
        exit(1);
    }

    return image;
}

// Linearize the RGB values. Images are typically stored in a non-linear 
// gamma corrected format, so we need to account for this.
float linearize(float channel) {
    if (channel <= 0.04045) {
        return channel / 12.92;
    } else {
        return pow((channel + 0.055) / 1.055, 2.4);
    }
}

cv::Vec3f rgbToXyz(cv::Vec3f rgb) { 
    float r = linearize(rgb[2]);
    float g = linearize(rgb[1]);
    float b = linearize(rgb[0]);

    // Convert from sRGB to the CIE XYZ color space using the D65 illuminant.
    // The conversion matrix is based on the D65 illuminant (a standard form of daylight).
    float X = r * 0.4124564 + g * 0.3575761 + b * 0.1804375;
    float Y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750;
    float Z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041;

    return cv::Vec3f(X, Y, Z);
}


cv::Vec3f xyzToLab(cv::Vec3f xyz) {
    // The reference white point for D65 illuminant in the XYZ color space.
    float ref_X =  0.95047; 
    float ref_Y =  1.00000;
    float ref_Z =  1.08883;

    // Check if the XYZ values are scaled between 0 to 255, and if so, normalize them to 0 to 1 range.
    if (xyz[0] > 1 || xyz[1] > 1 || xyz[2] > 1){
        xyz[0] = xyz[0] / 255.0f;
        xyz[1] = xyz[1] / 255.0f;
        xyz[2] = xyz[2] / 255.0f;
    }

    // Normalize the XYZ values with the reference white point.
    float x = xyz[0] / ref_X;
    float y = xyz[1] / ref_Y;
    float z = xyz[2] / ref_Z;

    // Convert XYZ to Lab. This involves a piecewise function for each coordinate.
    // The constants and equations come from the official Lab color space definition.
    x = (x > 0.008856) ? pow(x, 1.0/3.0) : (7.787 * x + 16.0/116.0);
    y = (y > 0.008856) ? pow(y, 1.0/3.0) : (7.787 * y + 16.0/116.0);
    z = (z > 0.008856) ? pow(z, 1.0/3.0) : (7.787 * z + 16.0/116.0);
    
    // Compute the L*, a*, and b* values from the transformed x, y, z coordinates.
    float L = (116.0 * y) - 16.0;
    float a = 500.0 * (x - y);
    float b = 200.0 * (y - z);

    return cv::Vec3f(L, a, b);
}



//ΔEab ​= sqrt(L2​−L1​)**2 + (a2​−a1​)**2 + (b2​−b1​)**2)
float compute_lab(cv::Vec3b pix_img_1, cv::Vec3b pix_img_2){
    cv::Vec3f xyz1 = rgbToXyz(pix_img_1);
    cv::Vec3f lab1 = xyzToLab(xyz1);    
    
    cv::Vec3f xyz2 = rgbToXyz(pix_img_2);
    cv::Vec3f lab2 = xyzToLab(xyz2);

    float L1 = lab1[0];
    float a1 = lab1[1];
    float b1 = lab1[2];

    float L2 = lab2[0];
    float a2 = lab2[1];
    float b2 = lab2[2];


    float luminance_diff = pow(L2-L1, 2);
    float a_diff = pow(a2-a1, 2);
    float b_diff = pow(b2-b1, 2);
    
    float lab = sqrt(luminance_diff + a_diff + b_diff);


    return lab;
}


cv::Mat compute_lab_image(cv::Mat image_t1, cv::Mat image_t2){
    cv::Mat lab_image(image_t1.rows, image_t1.cols, CV_32FC3);
    

    for (int i = 0; i < image_t1.rows; i++){
        for (int j = 0; j < image_t1.cols; j++){
            cv::Vec3b pixel_img_1 = image_t1.at<cv::Vec3b>(i, j);
            cv::Vec3b pixel_img_2 = image_t2.at<cv::Vec3b>(i, j);
            float lab = compute_lab(pixel_img_1, pixel_img_2);

            lab_image.at<cv::Vec3f>(i, j) = lab;
        }
    }

    return lab_image;
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


int main(int argc, char** argv){
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <path_to_input_image> <path_to_save_image>" << std::endl;
        return 1;
    }

    std::string path_image1 = argv[1];
    std::string path_image2 = argv[2];
    std::string path_save = argv[3];

    cv::Mat image1 = load_image(path_image1);
    cv::Mat image2 = load_image(path_image2);
    cv::Mat lab_image = compute_lab_image(image1, image2);
    save_image(path_save, lab_image);
    return 0;
}

//
//run with g++ lab_space.cpp -o lab_space `pkg-config --cflags --libs opencv4`
// ./lab_space "../../subject/bg.jpg" "../../subject/frame.jpg" "image_lab.jpg"
