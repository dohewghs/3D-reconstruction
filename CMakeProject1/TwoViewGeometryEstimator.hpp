#pragma once
#include "ITwoViewGeometryEstimator.hpp"

class TwoViewGeometryEstimator : public ITwoViewGeometryEstimator
{
private:
    // Private members and methods
    double m_ransacProb;
    double m_ransacThreshold;

    cv::Mat m_cameraMatrix;

public:
    TwoViewGeometryEstimator(const cv::Mat& cameraMatrix, 
                            double ransacProb = 0.999, 
                            double ransacThreshold = 1.0)
        : m_cameraMatrix(cameraMatrix.clone()), 
        m_ransacProb(ransacProb), 
        m_ransacThreshold(ransacThreshold) 
    {}
    
    TwoViewReconstructionResult reconstruct(const Frame& frame1, 
                                            const Frame& frame2, 
                                            const std::vector<cv::DMatch>& matches) override;

};