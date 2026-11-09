#pragma once 

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

struct Frame;

class IFeatureMatcher
{
public:
    virtual ~IFeatureMatcher() = default;
    virtual std::vector<cv::DMatch> match(const cv::Mat& descriptors1, const cv::Mat& descriptors2) = 0;
};