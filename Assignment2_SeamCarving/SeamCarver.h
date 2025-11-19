#pragma once
#include "opencv2/opencv.hpp"
#include <vector>
#define SOLUTION
#define USE_DP // Use dynamic programming for seam finding by default, otherwise use greedy approach

/**
 * @class SeamCarver
 * @brief Implements content-aware image resizing using the Seam Carving algorithm.
 *
 * This class provides functionality to reduce image width and/or height by
 * iteratively removing low-energy seams. The seams are determined using either:
 *   - Dynamic Programming (default), which finds the globally minimum-energy seam.
 *   - Greedy Algorithm (when USE_DP is disabled), which selects the locally best seam.
 */
class SeamCarver {
private:
    cv::Mat image; // Internal image storage containing current working image.

    /**
     * @brief Computes the energy map of the current image.
     * @return A single-channel floating-point matrix representing per-pixel energy.
     */
    cv::Mat computeEnergyMap();

    /**
     * @brief Finds a vertical seam of minimum energy.
     * A vertical seam is one pixel per row, connected from top to bottom.
     *
     * @param energy Pre-computed energy map of the image.
     * @return Vector of x-coordinates for each row indicating the seam position.
     */
    std::vector<int> findVerticalSeam(const cv::Mat& energy);

    /**
     * @brief Finds a horizontal seam of minimum energy.
     * A horizontal seam is one pixel per column, connected left to right.
     *
     * @param energy Pre-computed energy map.
     * @return Vector of y-coordinates for each column indicating the seam position.
     */
    std::vector<int> findHorizontalSeam(const cv::Mat& energy);

    /**
     * @brief Removes a vertical seam from the image.
     * Given a seam specifying one column index per row, this function shifts the
     * remaining pixels left and reduces the image width by one.
     *
     * @param seam Vector of x-coordinates specifying the seam path.
     */
    void removeVerticalSeam(const std::vector<int>& seam);

    /**
     * @brief Removes a horizontal seam from the image.
     * Given a seam specifying one row index per column, this function shifts remaining
     * pixels upward and reduces the image height by one.
     *
     * @param seam Vector of y-coordinates specifying the seam path.
     */
    void removeHorizontalSeam(const std::vector<int>& seam);

public:
    /**
     * @brief Constructs a new SeamCarver with an initial image.
     * The provided image is copied into internal storage and manipulated
     * during the seam carving process.
     *
     * @param img Input BGR image.
     */
    SeamCarver(const cv::Mat& img) : image(img) {};

    /**
     * @brief Resizes the image to a target width and height using seam carving.
     *
     * @param targetWidth  The desired final width of the image.
     * @param targetHeight The desired final height of the image.
     */
    void resize(int targetWidth, int targetHeight);

    /**
     * @brief Retrieves the current processed image.
     * @return The modified image after carving operations.
     */
    cv::Mat getImage() const { return image; };
};