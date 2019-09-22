#ifndef coins_identificator
#define coins_identificator

#include <opencv2/core.hpp>
#include <vector>
#include <opencv2/dnn.hpp>
#include <string>

class CoinsIdentificator
{
private:
	cv::dnn::Net cnn;
	std::vector<std::string> coin_id;
public:
	/*
	Identificator constructor: it requires the drive paths to the .pb (weights)
	and .pbtxt (net configuration) input files and the labels of the classifier (in order)
	*/
	CoinsIdentificator(
		const std::string &model_weights_path,
		const std::string &model_config_path,
		const std::vector<std::string> labels);

	/*
	Try to identify the coins: the values predicted with a confidence more than
	conf_thresh are considered unsure, with max_conf or more of these values the prediction
	is rejected. Then, return the found labels (empty string if coin not identified,
	corresponding string if found)
	*/
	std::vector<std::string> Identify(
		cv::Mat &input_img,
		std::vector<cv::Vec3f> coins,
		std::vector<cv::Vec3f> &evaluated_coins,
		float conf_thresh,
		int max_conf);
};

#endif // !coins_identificator