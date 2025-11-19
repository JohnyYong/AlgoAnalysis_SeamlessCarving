#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#define SOLUTION
//#define USE_DP // Use dynamic programming for seam finding by default, otherwise use greedy approach

class SeamCarver {
private:
    cv::Mat image;

    cv::Mat computeEnergyMap();
    std::vector<int> findVerticalSeam(const cv::Mat& energy);
    std::vector<int> findHorizontalSeam(const cv::Mat& energy);
    void removeVerticalSeam(const std::vector<int>& seam);
    void removeHorizontalSeam(const std::vector<int>& seam);

public:
    SeamCarver(const cv::Mat& img) : image(img) {};
    void resize(int targetWidth, int targetHeight);
    cv::Mat getImage() const { return image; };
};