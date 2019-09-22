#include "CoinsIdentificator.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

CoinsIdentificator::CoinsIdentificator(const std::string & model_weights_path, const std::string & model_config_path, const std::vector<std::string> labels)
{
	cnn = cv::dnn::readNetFromTensorflow(model_weights_path, model_config_path);
	coin_id = labels;
}

std::vector<std::string> CoinsIdentificator::Identify(cv::Mat & input_img, std::vector<cv::Vec3f> coins, std::vector<cv::Vec3f> & evaluated_coins, float confusion_thresh, int max_conf)
{
	std::vector<std::string> predictions;

	//Parse all input coins (vector of circles)
	for (int i = 0; i < coins.size(); i++)
	{
		cv::Mat coin;
		int radius = cvRound(coins[i][2]);

		try
		{
			//Region of interest is the coin circle
			cv::Rect roi(
				cvRound(coins[i][0]) - cvRound(coins[i][2]),
				cvRound(coins[i][1]) - cvRound(coins[i][2]),
				radius * 2,
				radius * 2);

			coin = input_img(roi);
		}
		/*Do not consider circles near the edges of the image:
		missclassification very likely in these regions!*/
		catch (cv::Exception ex) { continue; }

		//Process the image to get nn input
		cv::Mat cnn_input;
		cv::resize(coin, cnn_input, cv::Size(150, 150));
		cv::cvtColor(cnn_input, cnn_input, cv::COLOR_BGR2RGB);
		cnn_input.convertTo(cnn_input, CV_32FC3, 1.f / 255);

		cnn.setInput(cv::dnn::blobFromImage(cnn_input));

		cv::Mat prob = cnn.forward();

		//Find the prediction with the most confidence
		cv::Point classIdPoint;
		double max_confidence;
		minMaxLoc(prob.reshape(1, 1), 0, &max_confidence, 0, &classIdPoint);
		int max_classId = classIdPoint.x;

		//Reject prediction if unsure on max_conf or more values
		int conf = 0;

		for (int j = 0; j < prob.cols; j++)
		{
			if (prob.at<float>(0, j) != max_confidence && prob.at<float>(0, j) > confusion_thresh)
				conf++;
		}
		if (conf < max_conf)
			predictions.push_back(coin_id[max_classId]);
		else
			predictions.push_back("");

		evaluated_coins.push_back(coins[i]);
	}
	
	return predictions;
}
