#pragma once
#include "IFeatureExtractor.hpp"
#include "Types.hpp"

#include <vector>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

class AkazeFeatureExtractor : public IFeatureExtractor
{
private:
	cv::Ptr<cv::AKAZE> detector;
public:
    explicit AkazeFeatureExtractor(
        float threshold = 0.001f,           // Поріг контрастності (детектора)
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB, // Бінарний дескриптор
        int descriptorSize = 0,             // 0 = повний розмір (486 біт)
        int descriptorChannels = 3,         // 3 канали (яскравість + градієнти)
        int nOctaves = 4,                   // Кількість октав у просторі масштабів
        int nOctaveLayers = 4               // Кількість підрівнів на октаву
    );

    void extract(Frame& frame) override;
    int getDefaultNorm() const override;
    double getThreshold() const;
};