#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>

struct Frame
{
	int id;
	cv::Mat imageGray;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
};

struct ImagePairMatch
{
	int frameId1;
	int frameId2;
	std::vector<cv::DMatch> matches;
	cv::Mat inlierMask;
};

struct CameraPose
{
	Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
	Eigen::Vector3d t = Eigen::Vector3d::Zero();
};

struct LandMark3D
{
	int id;
	Eigen::Vector3d position;
	// Список спостережень: pair<frame_id, keypoint_idx> для Bundle Adjustment
	std::vector<std::pair<int, int>> observations;
};