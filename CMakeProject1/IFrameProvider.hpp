#pragma once
#include <opencv2/core.hpp>

class IFrameProvider
{
	virtual ~IFrameProvider() = default;

	virtual bool getNextFrame(cv::Mat& outFrame) = 0;
	virtual bool hasNextFrame() const = 0;
};


// VideoFileFrameProvider
// DirectoryFrameProvider
// LiveStreamFrameProvider