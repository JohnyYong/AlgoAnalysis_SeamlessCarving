#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>
#include "SeamCarver.h"
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/logger.hpp>   // make logger available
#include "SeamCarver.h"
#include <iostream>

#ifdef SOLUTION
int main(int argc, char** argv)
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    std::string path;
    int targetWidth;
    int targetHeight;

    //For input image and resolutions
    if (argc >= 2)
    {
        path = argv[1];
        std::cout << "Using input image from argument: " << path << std::endl;
    }
    else
    {
        std::cout << "Input the file path: ";
        std::cin >> path;
    }

    cv::Mat image = cv::imread(path);
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    std::cout << "Original size: " << image.size() << std::endl;

    if (argc >= 4)
    {
        targetWidth = std::stoi(argv[2]);
        targetHeight = std::stoi(argv[3]);
        std::cout << "Target size from arguments: "
            << targetWidth << " x " << targetHeight << std::endl;
    }
    else
    {
        std::cout << "Input target width: ";
        std::cin >> targetWidth;

        std::cout << "Input target height: ";
        std::cin >> targetHeight;
    }

    if (targetWidth <= 0 || targetHeight <= 0)
    {
        std::cerr << "Error: target width/height must be positive." << std::endl;
        return -1;
    }

    SeamCarver carver(image);
    carver.resize(targetWidth, targetHeight);

    // Save result
    cv::Mat result = carver.getImage();
    cv::imwrite("output.jpg", result);

    std::cout << "New size: " << result.size() << std::endl;
    std::cout << "Done! Saved to output.jpg" << std::endl;

    return 0;
}
#endif // SOLUTION

