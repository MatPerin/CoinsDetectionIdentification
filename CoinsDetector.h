#ifndef coins_detector
#define coins_detector

#include <opencv2/core.hpp>
#include <vector>

class CoinsDetector
{
private:

	//Class constructor
	CoinsDetector();

public:

	//PublicMethods
	/*
	Preprocess the image:
		1 - Rescale to 800 pixel height, maintaining scale
		2 - Convert to grayscale
		3 - Apply a 9x9 kernel with std 1 Gaussian Blur for
		noise reduction
	Compute the circles corresponding to coins in the image:
		1 - Rough Hough Transform (higher Canny threshold)
		2 - Remove outliers by constaining the circles to
		be close to the mean radius previously found
		3 - Find the circle with least variance in colour,
		this is the most likely ot be a coin
		4 - Apply a finer Hough Transform but constaining
		the radii to be close to the most likely one
		5 - Return found circles
	*/
	static std::vector<cv::Vec3f> Detect(cv::Mat &input_img, cv::Mat &output_img);
};

#endif // !coins_detector
