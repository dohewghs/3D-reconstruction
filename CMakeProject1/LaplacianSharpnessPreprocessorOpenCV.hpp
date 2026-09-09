#pragma once
#include "IFramePreprocessor.hpp"
#include <opencv2/imgproc.hpp>

class LaplacianFramePreprocessorOpenCV : public IFramePreprocessor
{
private:
	double blurThreshold;

public:
	explicit LaplacianFramePreprocessorOpenCV(double _blurThreshold = 100);
	~LaplacianFramePreprocessorOpenCV() = default;

	bool process(const cv::Mat& rawFrame, cv::Mat& outFrame) override;
	double calculateVarianceOfLaplacian(const cv::Mat& gray) const;

	void setThreshold(double threshold);
	double getThreshold() const;
};