#pragma once
#include <opencv2/core.hpp>
#include <Eigen/Dense>
#include <vector>
#include <unordered_map>

struct Frame
{
	int id;
	cv::Mat imageGray;
	cv::Mat imageColor; // Оригінальний BGR кадр (потрібен для кольору щільної хмари)
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
    int trackId = -1;   // Трек (ланцюжок збігів між кадрами), з якого тріангульовано точку
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
    int sourceId = -1; // Індекс карти глибин, з якої отримано точку
};

/**
 * Карта глибин одного кадру, отримана зі стерео-пари (refId, srcId).
 * depth(y, x) = Z у системі камери refId (0 = немає даних).
 */
struct DepthMap {
    int refId = -1;
    int srcId = -1;
    cv::Mat depth; // CV_32F
};
