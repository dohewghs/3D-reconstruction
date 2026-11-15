#pragma once

#include "IFrameProvider.hpp"
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <string>
#include <opencv2/videoio.hpp>

class VideoFrameProvider : public IFrameProvider
{
private:
    cv::VideoCapture cap;
    size_t frameStep;

    bool isFirstFrame = true;
public:
    VideoFrameProvider() = delete;
    explicit VideoFrameProvider(std::string videoFilePath, size_t frameStep = 1);

    virtual bool getNextFrame(cv::Mat& outFrame) override;
	virtual void reset() override;
};