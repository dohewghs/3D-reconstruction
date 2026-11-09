#pragma once

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
#include "Types.hpp"

class ITwoViewGeometry
{
public:
    virtual ~ITwoViewGeometry() = default;
    virtual TwoViewReconstructionResult reconstruct(const Frame& frame1, 
                                                    const Frame& frame2, 
                                                    const std::vector<cv::DMatch>& matches) = 0;
};