#include "VideoFrameProvider.hpp"
#include <opencv2/core/mat.hpp>
#include <stdexcept>
#include <iostream>

VideoFrameProvider::VideoFrameProvider(std::string videoFilePath, size_t frameStep)
    : cap(videoFilePath), frameStep(frameStep), isFirstFrame(true)
{
    if (!cap.isOpened())
    {
        throw std::runtime_error("Failed to open file: " + videoFilePath);
    }
}

bool VideoFrameProvider::getNextFrame(cv::Mat& outFrame)
{
    if (!isFirstFrame)
    {
        for (int i=0; i<frameStep - 1; ++i)
        {
            if (!cap.grab()) 
                return false;
            
        }
    }

    isFirstFrame = false;

    if (!cap.grab())
        return false;

    return cap.retrieve(outFrame);
}

void VideoFrameProvider::reset()
{
}