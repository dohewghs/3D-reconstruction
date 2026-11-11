#pragma once

#include "IFeatureMatcher.hpp"

/**
 * Brute-force зіставлення з тестом Лоу для будь-якої норми:
 * NORM_HAMMING для бінарних дескрипторів (AKAZE, ORB), NORM_L2 для SIFT.
 * Норму зручно брати з IFeatureExtractor::getDefaultNorm().
 */
class RatioMatcher : public IFeatureMatcher
{
private:
    float ratioThreshold;
    cv::Ptr<cv::BFMatcher> matcher;

public:
    explicit RatioMatcher(int normType, float ratioThreshold = 0.8f)
        : ratioThreshold(ratioThreshold)
    {
        matcher = cv::BFMatcher::create(normType, false);
    }

    std::vector<cv::DMatch> match(const cv::Mat& descriptors1, const cv::Mat& descriptors2) override
    {
        std::vector<std::vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descriptors1, descriptors2, knnMatches, 2);

        std::vector<cv::DMatch> goodMatches;
        for (const auto& matchPair : knnMatches)
        {
            if (matchPair.size() != 2) continue;
            if (matchPair[0].distance < ratioThreshold * matchPair[1].distance)
                goodMatches.push_back(matchPair[0]);
        }
        return goodMatches;
    }
};
