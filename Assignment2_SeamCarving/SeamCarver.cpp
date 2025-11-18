//// SeamCarver.cpp
#include "SeamCarver.h"
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <limits>
#include <iomanip>

#ifdef SOLUTION

//Time Variables 
namespace {
    double total_energy_time = 0.0;
    double total_vertical_seam_time = 0.0;
    double total_horizontal_seam_time = 0.0;
    double total_vertical_remove_time = 0.0;
    double total_horizontal_remove_time = 0.0;
    int energy_calls = 0;
    int vertical_seam_calls = 0;
    int horizontal_seam_calls = 0;
    int vertical_remove_calls = 0;
    int horizontal_remove_calls = 0;
}

//Function to print timing results
void printTimingResults() {
    std::cout << "\n===TimeTest===" << std::endl;
    std::cout << "Energy Calculation:" << std::endl;
    std::cout << "  Calls: " << energy_calls << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(4) << total_energy_time << "s" << std::endl;
    std::cout << "  Average time: " << (energy_calls > 0 ? total_energy_time / energy_calls : 0) << "s" << std::endl;

    std::cout << "\nVertical Seam Finding:" << std::endl;
    std::cout << "  Calls: " << vertical_seam_calls << std::endl;
    std::cout << "  Total time: " << total_vertical_seam_time << "s" << std::endl;
    std::cout << "  Average time: " << (vertical_seam_calls > 0 ? total_vertical_seam_time / vertical_seam_calls : 0) << "s" << std::endl;

    std::cout << "\nHorizontal Seam Finding:" << std::endl;
    std::cout << "  Calls: " << horizontal_seam_calls << std::endl;
    std::cout << "  Total time: " << total_horizontal_seam_time << "s" << std::endl;
    std::cout << "  Average time: " << (horizontal_seam_calls > 0 ? total_horizontal_seam_time / horizontal_seam_calls : 0) << "s" << std::endl;

    std::cout << "\nVertical Seam Removal:" << std::endl;
    std::cout << "  Calls: " << vertical_remove_calls << std::endl;
    std::cout << "  Total time: " << total_vertical_remove_time << "s" << std::endl;
    std::cout << "  Average time: " << (vertical_remove_calls > 0 ? total_vertical_remove_time / vertical_remove_calls : 0) << "s" << std::endl;

    std::cout << "\nHorizontal Seam Removal:" << std::endl;
    std::cout << "  Calls: " << horizontal_remove_calls << std::endl;
    std::cout << "  Total time: " << total_horizontal_remove_time << "s" << std::endl;
    std::cout << "  Average time: " << (horizontal_remove_calls > 0 ? total_horizontal_remove_time / horizontal_remove_calls : 0) << "s" << std::endl;

    double total_processing_time = total_energy_time + total_vertical_seam_time + total_horizontal_seam_time +
        total_vertical_remove_time + total_horizontal_remove_time;
    std::cout << "\nTOTAL PROCESSING TIME: " << total_processing_time << "s" << std::endl;
}

static void drawSeamOnImage(cv::Mat& img, const std::vector<int>& seam, bool vertical)
{
    //Color for the visualisation
    cv::Vec3b color(0, 0, 255);

    if (vertical)
    {

        int rows = img.rows;
        int cols = img.cols;
        for (int y = 0; y < rows && y < (int)seam.size(); ++y)
        {
            int x = seam[y];
            if (x >= 0 && x < cols)
            {
                // If grayscale, convert to BGR first
                if (img.channels() == 1)
                {
                    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
                }
                img.at<cv::Vec3b>(y, x) = color;
            }
        }
    }
    else //For horizontal seam
    {
       
        int rows = img.rows;
        int cols = img.cols;
        for (int x = 0; x < cols && x < (int)seam.size(); ++x)
        {
            int y = seam[x];
            if (y >= 0 && y < rows)
            {
                if (img.channels() == 1)
                {
                    cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
                }
                img.at<cv::Vec3b>(y, x) = color;
            }
        }
    }
}

// Reset timing counters
void resetTiming() {
    total_energy_time = 0.0;
    total_vertical_seam_time = 0.0;
    total_horizontal_seam_time = 0.0;
    total_vertical_remove_time = 0.0;
    total_horizontal_remove_time = 0.0;
    energy_calls = 0;
    vertical_seam_calls = 0;
    horizontal_seam_calls = 0;
    vertical_remove_calls = 0;
    horizontal_remove_calls = 0;
}

cv::Mat SeamCarver::computeEnergyMap()
{
    auto start = std::chrono::high_resolution_clock::now();

    //convert image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);

    // use Sobel to calculate the gradient of the image in the x and y direction
    cv::Mat grayX, grayY;
    cv::Sobel(gray, grayX, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grayY, CV_32F, 0, 1, 3);

    cv::Mat energy;
    // compute the energy map as the sum of the absolute values of the gradients
    cv::magnitude(grayX, grayY, energy);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    total_energy_time += duration.count();
    energy_calls++;

    return energy;
}

// Find minimum vertical seam
std::vector<int> SeamCarver::findVerticalSeam(const cv::Mat& energy)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int rows = energy.rows;
    const int cols = energy.cols;

    if (rows == 0 || cols == 0)
        return std::vector<int>();

    if (rows == 1) {
        double minVal;
        cv::Point minLoc;
        cv::minMaxLoc(energy.row(0), &minVal, nullptr, &minLoc, nullptr);
        return std::vector<int>(1, minLoc.x);
    }

    const int size = rows * cols;

    // DP and parent in 1D (row-major)
    std::vector<float> dp(size);
    std::vector<int> parent(size, -1);

    //Lambda to do 1D array instead of vector
    auto idx = [cols](int r, int c) { return r * cols + c; };

    // first row: dp(0,x) = energy(0,x)
    const float* eRow0 = energy.ptr<float>(0);
    for (int x = 0; x < cols; ++x) {
        dp[idx(0, x)] = eRow0[x];
    }

    constexpr float INF = std::numeric_limits<float>::infinity();

    // DP: from second row to last
    for (int y = 1; y < rows; ++y) {
        const float* eRow = energy.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            // candidates from row y-1
            float bestCost = dp[idx(y - 1, x)];
            int   bestX = x;

            if (x > 0 && dp[idx(y - 1, x - 1)] < bestCost) {
                bestCost = dp[idx(y - 1, x - 1)];
                bestX = x - 1;
            }
            if (x + 1 < cols && dp[idx(y - 1, x + 1)] < bestCost) {
                bestCost = dp[idx(y - 1, x + 1)];
                bestX = x + 1;
            }

            dp[idx(y, x)] = eRow[x] + bestCost;
            parent[idx(y, x)] = bestX;
        }
    }

    // find minimum in last row
    float minCost = dp[idx(rows - 1, 0)];
    int minIndex = 0;
    for (int x = 1; x < cols; ++x) {
        float v = dp[idx(rows - 1, x)];
        if (v < minCost) {
            minCost = v;
            minIndex = x;
        }
    }

    // backtrack seam
    std::vector<int> seam(rows);
    int x = minIndex;
    for (int y = rows - 1; y >= 0; --y) {
        seam[y] = x;
        int p = parent[idx(y, x)];
        x = p;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    total_vertical_seam_time += duration.count();
    vertical_seam_calls++;

    return seam;
}

std::vector<int> SeamCarver::findHorizontalSeam(const cv::Mat& energy)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int rows = energy.rows;
    const int cols = energy.cols;

    if (rows == 0 || cols == 0)
        return std::vector<int>();

    if (cols == 1) {
        double minVal;
        cv::Point minLoc;
        cv::minMaxLoc(energy.col(0), &minVal, nullptr, &minLoc, nullptr);
        return std::vector<int>(1, minLoc.y);
    }

    const int size = rows * cols;

    std::vector<float> dp(size);
    std::vector<int> parent(size, -1);

    //Lambda to do 1D array instead of vector
    auto idx = [cols](int r, int c) { return r * cols + c; };

    for (int y = 0; y < rows; ++y) {
        const float* eRow = energy.ptr<float>(y);
        dp[idx(y, 0)] = eRow[0];
    }

    for (int x = 1; x < cols; ++x) {
        for (int y = 0; y < rows; ++y) {
            const float* eRow = energy.ptr<float>(y);

            float bestCost = dp[idx(y, x - 1)];
            int   bestY    = y;

            if (y > 0 && dp[idx(y - 1, x - 1)] < bestCost) {
                bestCost = dp[idx(y - 1, x - 1)];
                bestY    = y - 1;
            }
            if (y + 1 < rows && dp[idx(y + 1, x - 1)] < bestCost) {
                bestCost = dp[idx(y + 1, x - 1)];
                bestY    = y + 1;
            }

            dp[idx(y, x)]      = eRow[x] + bestCost;
            parent[idx(y, x)]  = bestY;
        }
    }

    // find minimum in last column
    float minCost = dp[idx(0, cols - 1)];
    int minIndex  = 0;
    for (int y = 1; y < rows; ++y) {
        float v = dp[idx(y, cols - 1)];
        if (v < minCost) {
            minCost = v;
            minIndex = y;
        }
    }

    // backtrack seam (one y index per column)
    std::vector<int> seam(cols);
    int y = minIndex;
    for (int x = cols - 1; x >= 0; --x) {
        seam[x] = y;
        int p = parent[idx(y, x)];
        y = p;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    total_horizontal_seam_time += duration.count();
    horizontal_seam_calls++;

    return seam;
}

void SeamCarver::removeVerticalSeam(const std::vector<int>& seam)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int rows = image.rows;
    const int cols = image.cols;

    // Create a new image with one less column
    cv::Mat newImage(rows, cols - 1, image.type());

    // Remove the seam
    for (int i = 0; i < rows; ++i)
    {
        int seamCol = seam[i];

        // Copy all rows before the seam
        if (seamCol > 0)
            image(cv::Range(i, i + 1), cv::Range(0, seamCol)).copyTo(newImage(cv::Range(i, i + 1), cv::Range(0, seamCol)));

        // Copy all rows after the seam
        if (seamCol < cols - 1)
            image(cv::Range(i, i + 1), cv::Range(seamCol + 1, cols)).copyTo(newImage(cv::Range(i, i + 1), cv::Range(seamCol, cols - 1)));
    }

    image = newImage;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    total_vertical_remove_time += duration.count();
    vertical_remove_calls++;
}

void SeamCarver::removeHorizontalSeam(const std::vector<int>& seam)
{
    auto start = std::chrono::high_resolution_clock::now();

    const int rows = image.rows;
    const int cols = image.cols;

    // Create a new image with one less row
    cv::Mat newImage(rows - 1, cols, image.type());

    // Remove the seam
    for (int i = 0; i < cols; ++i)
    {
        int seamRow = seam[i];

        // Copy all rows before the seam
        if (seamRow > 0)
            image(cv::Range(0, seamRow), cv::Range(i, i + 1)).copyTo(newImage(cv::Range(0, seamRow), cv::Range(i, i + 1)));

        // Copy all rows after the seam
        if (seamRow < rows - 1)
            image(cv::Range(seamRow + 1, rows), cv::Range(i, i + 1)).copyTo(newImage(cv::Range(seamRow, rows - 1), cv::Range(i, i + 1)));
    }

    image = newImage;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    total_horizontal_remove_time += duration.count();
    horizontal_remove_calls++;
}

void SeamCarver::resize(int targetWidth, int targetHeight)
{
    // Reset timing at the start of each resize operation
    resetTiming();

    auto total_start = std::chrono::high_resolution_clock::now();

    if (targetWidth <= 0 || targetHeight <= 0)
    {
        std::cerr << "Invalid target size\n";
        return;
    }

    if (targetWidth >= image.cols && targetHeight >= image.rows)
    {
        std::cerr << "Invalid target size\n";
        return;
    }

    std::cout << "Starting seam carving from " << image.cols << "x" << image.rows
        << " to " << targetWidth << "x" << targetHeight << std::endl;
    std::cout << "Removing " << (image.cols - targetWidth) << " vertical and "
        << (image.rows - targetHeight) << " horizontal seams..." << std::endl;

    int iteration = 0;
    while (image.cols > targetWidth || image.rows > targetHeight)
    {
        int removeWidth = image.cols - targetWidth; //remaining vertical seams to remove
        int removeHeight = image.rows - targetHeight; //remaining horizontal seams to remove

        //Check dimensions that require shrinking
        bool removeVert = false;
        if (removeWidth > 0 && removeHeight == 0)
            removeVert = true;
        else if (removeHeight > 0 && removeWidth == 0)
            removeVert = false;
        else
        {
            //choose the dimension with larger normalized remaining fraction
            double fracW = static_cast<double>(removeWidth) / image.cols;
            double fracH = static_cast<double>(removeHeight) / image.rows;
            removeVert = (fracW >= fracH);
        }

        if (removeVert && removeWidth > 0)
        {
            cv::Mat energy = computeEnergyMap();
            std::vector<int> seam = findVerticalSeam(energy);

            cv::Mat vis = image.clone();
            drawSeamOnImage(vis, seam, true);

#ifndef VISUALISE
            cv::imshow("SeamCarving (Also Vertical Visualisation)", vis);
            cv::waitKey(1);
#endif

            removeVerticalSeam(seam);
        }
        else if (!removeVert && removeHeight > 0)
        {
            cv::Mat energy = computeEnergyMap();
            std::vector<int> seam = findHorizontalSeam(energy);

            cv::Mat vis = image.clone();
            drawSeamOnImage(vis, seam, false);

#ifndef VISUALISE
            cv::imshow("Horizontal", vis);
            cv::waitKey(1);
#endif
            removeHorizontalSeam(seam);
        }
        else
        {
            break;
        }

        iteration++;
        //To display resizing progress
        if (iteration % 10 == 0) {
            std::cout << "Progress: " << iteration << " seams removed. Current size: "
                << image.cols << "x" << image.rows << std::endl;
        }
#ifndef VISUALISE
        cv::imshow("SeamCarving", image);
        cv::waitKey(1);
#endif
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = total_end - total_start;

    // Print detailed timing results
    printTimingResults();
    std::cout << "\nTOTAL WALL CLOCK TIME: " << std::fixed << std::setprecision(4)
        << total_duration.count() << "s" << std::endl;
    std::cout << "Final image size: " << image.cols << "x" << image.rows << std::endl;
}
#endif // SOLUTION