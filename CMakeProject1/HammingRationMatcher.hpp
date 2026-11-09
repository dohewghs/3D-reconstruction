#pragma once

#include "IFeatureMatcher.hpp"

class HammingRatioMatcher : public IFeatureMatcher
{
private:
    float ratioThreshold;
    cv::Ptr<cv::DescriptorMatcher> matcher;

public:
    HammingRatioMatcher(float ratioThreshold = 0.75f)
        : ratioThreshold(ratioThreshold)
    {
            this->matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    }

    std::vector<cv::DMatch> match(const cv::Mat& descriptors1, const cv::Mat& descriptors2) override
    {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        this->matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;

        for (const auto& matchPair : knnMatches)
        {
            if (matchPair.size() != 2)
                continue;

            if (matchPair[0].distance < ratioThreshold * matchPair[1].distance)
            {
                goodMatches.push_back(matchPair[0]);
            }
        }
        
        return goodMatches;
    }
};