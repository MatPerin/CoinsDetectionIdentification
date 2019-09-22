#include "CoinsDetector.h";
#include "CoinsIdentificator.h";
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <iostream>

const std::string model_path = "model.pb";
const std::string model_conf_path = "model.pbtxt";
const std::vector<std::string> labels = { "10c", "1c", "1e", "20c", "2c", "2e", "50c", "5c" };

int main()
{
	//Get input image path
	std::string img_path;
	std::cout << "Input image path: ";
	std::getline(std::cin, img_path);
	
	//Read input image
	cv::Mat img;
	try
	{
		img = cv::imread(img_path); 
		if (img.cols == 0 && img.rows == 0)
			throw cv::Exception();
	}
	catch (cv::Exception ex)
	{
		std::cout << "Cannot open specified image" << std::endl;
		return -1;
	}

	//Detect the coins in the image
	cv::Mat detected_img;
	std::vector<cv::Vec3f> coins;
	coins = CoinsDetector::Detect(img, detected_img);

	//Show detected coins
	cv::Mat detection_img;
	detected_img.copyTo(detection_img);
	for (int i = 0; i < coins.size(); i++)
	{
		cv::Point center(cvRound(coins[i][0]), cvRound(coins[i][1]));
		int radius = cvRound(coins[i][2]);
		circle(detection_img, center, radius, cv::Scalar(0, 255, 0), 2, 8, 0);
	}

	//Identify the coins
	CoinsIdentificator identificator(model_path, model_conf_path, labels);
	std::vector<cv::Vec3f> evaluated_coins;

	std::vector<std::string> predictions = identificator.Identify(detected_img, coins, evaluated_coins, 1e-5, 3);

	//Show identified coins and their relative value
	cv::Mat prediction_img;
	detected_img.copyTo(prediction_img);
	for (int i = 0; i < evaluated_coins.size(); i++)
	{
		cv::Point center(cvRound(evaluated_coins[i][0]), cvRound(evaluated_coins[i][1]));
		int radius = cvRound(evaluated_coins[i][2]);

		if(predictions[i] == "")
			circle(prediction_img, center, radius, cv::Scalar(0, 0, 255), 2, 8, 0);
		else
		{
			circle(prediction_img, center, radius, cv::Scalar(0, 255, 0), 2, 8, 0);
			cv::putText(
				prediction_img,
				predictions[i],
				cv::Point(cvRound(evaluated_coins[i][0]) - 40, cvRound(evaluated_coins[i][1]) + 15),
				cv::FONT_HERSHEY_COMPLEX,
				1,
				cv::Scalar(0, 255, 0),
				2,
				16);
		}
	}

	//Find image file name from path
	size_t found = img_path.find_last_of("/\\");
	std::string name = img_path.substr(found + 1);
	
	//Save results to disk
	cv::imwrite("detection_" + name, detection_img);
	cv::imwrite("identification_" + name, prediction_img);

	return 0;
}