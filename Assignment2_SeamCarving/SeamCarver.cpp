// SeamCarver.cpp
#include "SeamCarver.h"
#include <opencv2/imgproc.hpp>
#include <limits>

cv::Mat SeamCarver::computeEnergyMap()
{

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

	return energy;
}

// Find minimum vertical seam
std::vector<int> SeamCarver::findVerticalSeam(const cv::Mat& energy)
{
	const int rows = energy.rows;
	const int cols = energy.cols;

	// Edge cases
	if (cols == 0 || rows == 0)
	{
		return std::vector<int>();
	}
	if (rows == 1)
	{
		double minVal;
		cv::Point minLoc;
		cv::minMaxLoc(energy.row(0), &minVal, nullptr, &minLoc, nullptr);
		return std::vector<int>(1, minLoc.x);
	}

	//initialize cost matrix
	std::vector<std::vector<float>> dp(rows, std::vector<float>(cols, std::numeric_limits<float>::infinity()));

	//initialize parent matrix to reconstruct seam path
	std::vector<std::vector<int>> parent(rows, std::vector<int>(cols, -1));

	//fill first row of DP array
	for (int x = 0; x < cols; ++x)
		dp[0][x] = energy.at<float>(0, x);

	//fill DP and parent arrays
	for (int y = 1; y < rows; y++)
	{
		for (int x = 0; x < cols; x++)
		{
			float left = (x > 0) ? dp[y - 1][x - 1] : std::numeric_limits<float>::infinity();
			float mid = dp[y - 1][x];
			float right = (x < cols - 1) ? dp[y - 1][x + 1] : std::numeric_limits<float>::infinity();

			float minVal = std::min({ left, mid, right });
			dp[y][x] = energy.at<float>(y, x) + minVal;

			if (minVal == left)
				parent[y][x] = x - 1;
			else if (minVal == mid)
				parent[y][x] = x;
			else
				parent[y][x] = x + 1;
		}
	}

	//find min value in last row since it represents the end of the seam
	float minCost = dp[rows - 1][0];
	int minIndex = 0;
	for (int x = 1; x < cols; ++x)
	{
		if (dp[rows - 1][x] < minCost)
		{
			minCost = dp[rows - 1][x];
			minIndex = x;
		}
	}

	//reconstruct seam path
	std::vector<int> seam(rows);
	for (int y = rows - 1; y >= 0; --y)
	{
		seam[y] = minIndex;
		minIndex = parent[y][minIndex];
	}

	return seam;
}

// Find minimum horizontal seam
std::vector<int> SeamCarver::findHorizontalSeam(const cv::Mat& energy)
{
	const int rows = energy.rows;
	const int cols = energy.cols;

	// Edge cases
	if (cols == 0 || rows == 0)
	{
		return std::vector<int>();
	}
	if (cols == 1)
	{
		double minVal;
		cv::Point minLoc;
		cv::minMaxLoc(energy.col(0), &minVal, nullptr, &minLoc, nullptr);
		return std::vector<int>(1, minLoc.y);
	}

	//initialize cost matrix
	std::vector<std::vector<float>> dp(rows, std::vector<float>(cols, std::numeric_limits<float>::infinity()));

	//initialize parent matrix to reconstruct seam path
	std::vector<std::vector<int>> parent(rows, std::vector<int>(cols, -1));

	//fill first column of DP array
	for (int y = 0; y < rows; ++y)
		dp[y][0] = energy.at<float>(y, 0);

	//fill DP and parent arrays
	for (int x = 1; x < cols; ++x)
	{
		for (int y = 0; y < rows; ++y)
		{

			float upleft = (y - 1 >= 0) ? dp[y - 1][x - 1] : std::numeric_limits<float>::infinity();
			float middle = dp[y][x - 1];
			float downleft = (y + 1 < rows) ? dp[y + 1][x - 1] : std::numeric_limits<float>::infinity();

			float minVal = std::min({ upleft, middle, downleft });
			dp[y][x] = energy.at<float>(y, x) + minVal;

			if (minVal == upleft)
				parent[y][x] = y - 1;
			else if (minVal == middle)
				parent[y][x] = y;
			else
				parent[y][x] = y + 1;
		}
	}

	//find min value in last row since it represents the end of the seam
	float minCost = dp[0][cols - 1];
	int minIndex = 0;
	for (int y = 1; y < rows; ++y)
	{
		if (dp[y][cols - 1] < minCost)
		{
			minCost = dp[y][cols - 1];
			minIndex = y;
		}
	}

	//reconstruct seam path
	std::vector<int> seam(cols);
	for (int x = cols - 1; x >= 0; --x)
	{
		seam[x] = minIndex;
		minIndex = parent[minIndex][x];
	}

	return seam;
}

void SeamCarver::removeVerticalSeam(const std::vector<int>& seam)
{
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
}

void SeamCarver::removeHorizontalSeam(const std::vector<int>& seam)
{
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
}

void SeamCarver::resize(int targetWidth, int targetHeight)
{
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

	// Interleaved seam removal for better visual results
	while (image.cols > targetWidth || image.rows > targetHeight)
	{
		int removeWidth = image.cols - targetWidth; // remaining vertical seams to remove
		int removeHeight = image.rows - targetHeight; // remaining horizontal seams to remove

		// If only one dimension needs shrinking, pick that. Otherwise choose by ratio.
		bool removeVert = false;
		if (removeWidth > 0 && removeHeight == 0)
			removeVert = true;
		else if (removeHeight > 0 && removeWidth == 0)
			removeVert = false;
		else
		{
			// choose the dimension with larger normalized remaining fraction
			double fracW = static_cast<double>(removeWidth) / image.cols;
			double fracH = static_cast<double>(removeHeight) / image.rows;
			removeVert = (fracW >= fracH);
		}

		if (removeVert && removeWidth > 0)
		{
			cv::Mat energy = computeEnergyMap();
			std::vector<int> seam = findVerticalSeam(energy);
			removeVerticalSeam(seam);
		}
		else if (!removeVert && removeHeight > 0)
		{
			cv::Mat energy = computeEnergyMap();
			std::vector<int> seam = findHorizontalSeam(energy);
			removeHorizontalSeam(seam);
		}
		else
		{
			break;
		}

		cv::imshow("window", image);
		cv::waitKey(1);
	}
}
