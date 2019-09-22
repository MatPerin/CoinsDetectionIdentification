#include "CoinsDetector.h"
#include <opencv2/imgproc.hpp>

CoinsDetector::CoinsDetector(){}

std::vector<cv::Vec3f> CoinsDetector::Detect(cv::Mat & input_img, cv::Mat & output_img)
{
	//Rescale image (circle detection works better with smaller images)
	double ratio = (double)input_img.cols / input_img.rows;
	cv::resize(input_img, output_img, cv::Size(800, (int)800 / ratio));

	cv::Mat processed_image;

	//Convert to grayscale (for Canny edge detector)
	cv::cvtColor(output_img, processed_image, cv::COLOR_BGR2GRAY);

	//Apply gaussian filter (Hough Transform is sensitive to noise)
	cv::GaussianBlur(processed_image, processed_image, cv::Size(9, 9), 1, 1);

	std::vector<cv::Vec3f> circles;

	//Detect circles with Generalized Hough Transform
	/*
		Hough Circles output -> array of 3D vectors with:
		X coordinate of the center at index 0
		Y coordinate of the center at index 1
		Radius of the circle at index 2
	*/
	HoughCircles(processed_image,
		circles,
		cv::HOUGH_GRADIENT,
		1,
		processed_image.rows / 8,
		200,
		80,
		0,
		0);

	//Compute the mean radius
	double mean_radius = 0;
	for (int i = 0; i < circles.size(); i++)
		mean_radius += cvRound(circles[i][2]);
	mean_radius /= circles.size();

	//Remove outliers
	HoughCircles(
		processed_image,
		circles,
		cv::HOUGH_GRADIENT,
		1,
		processed_image.rows / 10,
		200,
		100,
		mean_radius - 100,
		mean_radius + 100);

	//Find the area with the least hue variance (costant colour)
	double best_index = 0;
	double best_variance = DBL_MAX;

	for (int i = 0; i < circles.size(); i++)
	{
		cv::Mat area;
		int radius = cvRound(circles[i][2]);

		try
		{
			//Get the region of interest (circle)
			cv::Rect roi(
				cvRound(circles[i][0]) - cvRound(circles[i][2]),
				cvRound(circles[i][1]) - cvRound(circles[i][2]),
				radius * 2,
				radius * 2);

			//Consider only the region of the circle
			area = output_img(roi);
		}
		//Do not consider circles close to the borders
		catch (cv::Exception ex) { continue; }

		//Convert the target area to HSV colour space
		cv::Mat area_hsv;
		cv::cvtColor(area, area_hsv, cv::COLOR_BGR2HSV);

		//Get the mean hue
		double mean = 0;
		for (int j = 0; j < area_hsv.rows; j++)
			for (int k = 0; k < area_hsv.cols; k++)
				mean += area_hsv.at<cv::Vec3b>(j, k)[0];
		mean /= (area_hsv.cols * area_hsv.rows);

		//Get the mean variance
		double variance = 0;
		for (int j = 0; j < area_hsv.rows; j++)
			for (int k = 0; k < area_hsv.cols; k++)
				variance += std::pow((area_hsv.at<cv::Vec3b>(j, k)[0] - mean), 2);
		variance /= (area_hsv.cols * area_hsv.rows);

		//Find the minimum hue variance among all circles
		if (variance < best_variance)
			best_index = i;
	}
	
	/*Most likely coin radius is the one of the area with
	the most constant colour: other coins should  have a
	similar radius*/
	int best_radius;
	if (circles.size() != 0)
		best_radius = cvRound(circles[best_index][2]);
	else
		return circles;

	int min_radius = best_radius - 17;
	int max_radius = best_radius + 17;

	//Recompute finer Hough Transform with radii constraint
	HoughCircles(
		processed_image,
		circles,
		cv::HOUGH_GRADIENT,
		1,
		min_radius,
		170,
		43,
		min_radius,
		max_radius);

	return circles;
}
