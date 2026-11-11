#pragma once
#include "IFeatureExtractor.hpp"
#include "Types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

/**
 * SIFT (у основному модулі features2d з OpenCV 4.4).
 * На нічних фото фасадів з повторюваними вікнами дає стабільніші збіги між
 * знімками з різних ракурсів, ніж бінарні дескриптори.
 */
class SiftFeatureExtractor : public IFeatureExtractor
{
private:
	cv::Ptr<cv::SIFT> detector;
public:
	explicit SiftFeatureExtractor(int maxFeatures = 8000, double contrastThreshold = 0.04)
	{
		detector = cv::SIFT::create(maxFeatures, 3, contrastThreshold);
	}

	void extract(Frame& frame) override
	{
		if (frame.imageGray.empty()) return;

		frame.keypoints.clear();
		frame.descriptors.release();
		detector->detectAndCompute(frame.imageGray, cv::noArray(), frame.keypoints, frame.descriptors);
	}

	int getDefaultNorm() const override
	{
		return cv::NORM_L2;
	}
};
