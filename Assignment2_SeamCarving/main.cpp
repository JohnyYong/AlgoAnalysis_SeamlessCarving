#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "SeamCarver.h"
#include <iostream>

#ifdef SOLUTION
int main() {
    // Suppress INFO warnings
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    std::string path;
    std::cout << "Input the file path: ";
    std::cin >> path;

    // Load image
    cv::Mat image = cv::imread(path);
    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    int width, height;
    std::cout << "Original size: " << image.size() << std::endl;
    std::cout << "Input the new width lower than " << image.size().width << " : ";
    std::cin >> width;

    std::cout << "Input the new height lower than " << image.size().height << " : ";
    std::cin >> height;


    // Create seam carver and resize
    SeamCarver carver(image);
    carver.resize(width, height);

    // Save result
    cv::Mat result = carver.getImage();
    cv::imwrite("output.jpg", result);

    std::cout << "New size: " << result.size() << std::endl;
    std::cout << "Done! Saved to output.jpg" << std::endl;

    return 0;
}
#endif // SOLUTION
