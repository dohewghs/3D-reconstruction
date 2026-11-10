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
	cv::Mat rotation;
	cv::Mat translation;

	CameraPose(cv::Mat rot = cv::Mat(), cv::Mat trans = cv::Mat())
        : rotation(std::move(rot)), translation(std::move(trans))
    {
    }
};

struct MapPoint3D {
    int id = -1;
    cv::Point3d position;
    cv::Mat descriptor; // Дескриптор, який описує точку

    std::unordered_map<int, int> observations; // <frame_id, keypoint_idx>

	bool isObservedIn(int frameId) const {
		return observations.find(frameId) != observations.end();
	}

	int getKeypointIndex(int frameId) const {
		auto it = observations.find(frameId);

		if (it != observations.end()) 
			return it->second;
		
		return -1; // Якщо кадр не спостерігає точку
	}
};

struct TwoViewReconstructionResult {
    bool success = false;
    cv::Mat R;
    cv::Mat t;
    std::vector<cv::Point3d> points3D;
    std::vector<cv::Point2f> inliersPrev;
    std::vector<cv::Point2f> inliersCurr;
    cv::Mat inlierMask;
	std::vector<int> inlierMatchIndices;
};

struct PnpTrackingResult {
	bool success = false;
	cv::Mat rotation; // Rotation matrix (3x3) world
	cv::Mat translation; // Translation vector (3x1) world
	int inliersCount = 0;

	std::vector<int> inlierPoints3D; // indices of inlier 3D points in the provided list
};

struct DensePoint3D {
    cv::Point3d position;
    cv::Vec3b color; // B, G, R
};