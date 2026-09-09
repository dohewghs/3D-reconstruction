#pragma once
#include <opencv2/core.hpp>

class IFramePreprocessor
{
public:
	virtual ~IFramePreprocessor() = default;

	virtual bool process(const cv::Mat& rawFrame, cv::Mat& outFrame) = 0;
};

// LaplacianSharpnessPreprocessor