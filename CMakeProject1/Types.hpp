#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>

struct Frame
{
	int id;
	cv::Mat imageGray;
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors;
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

struct TwoViewReconstructionResult {
    bool success = false;
    cv::Mat R;
    cv::Mat t;
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Point2f> inliersPrev;
    std::vector<cv::Point2f> inliersCurr;
    cv::Mat inlierMask;
};