#include "lab_space.hh"
#include "remove_noise.hh"
#include "tools.hh"


int main(int argc, char** argv){
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_input_image> <path_to_save_image>" << std::endl;
        return 1;
    }

    std::string path_image1 = argv[1];
    std::string path_image2 = argv[2];
    std::string output = "images_output/";

    cv::Mat image1 = load_image(path_image1);
    cv::Mat image2 = load_image(path_image2);
    
    cv::Mat lab_image = compute_lab_image(image1, image2);
    save_image(output + "image_lab.jpg", lab_image);

    cv::Mat opened_image = opening(lab_image, 3);
    save_image(output + "image_denoised.jpg", opened_image);

    return 0;
}