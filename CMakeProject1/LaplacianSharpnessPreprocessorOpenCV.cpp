#include "LaplacianSharpnessPreprocessorOpenCV.hpp"

LaplacianFramePreprocessorOpenCV::LaplacianFramePreprocessorOpenCV(double _blurThreshold) : blurThreshold(_blurThreshold)
{
}

bool LaplacianFramePreprocessorOpenCV::process(const cv::Mat& rawFrame, cv::Mat& outFrame)
{
	if (rawFrame.empty())
		return false;

	if (rawFrame.channels() == 3)
	{
		cv::cvtColor(rawFrame, outFrame, cv::COLOR_BGR2GRAY);
	}
	else if (rawFrame.channels() == 1)
	{
		outFrame = rawFrame;
	}
	else
	{
		cv::cvtColor(rawFrame, outFrame, cv::COLOR_BGRA2GRAY);
	}

	double score = this->calculateVarianceOfLaplacian(outFrame);

	if (score < this->blurThreshold)
		return false;

	return true;
}

double LaplacianFramePreprocessorOpenCV::calculateVarianceOfLaplacian(const cv::Mat& gray) const
{
	cv::Mat laplacian;

	cv::Laplacian(gray, laplacian, CV_64F);

	cv::Scalar mean, stddev;
	cv::meanStdDev(laplacian, mean, stddev);
	
	return stddev[0] * stddev[0];
}

void LaplacianFramePreprocessorOpenCV::setThreshold(double threshold)
{
	this->blurThreshold = threshold;
}

double LaplacianFramePreprocessorOpenCV::getThreshold() const
{
	return this->blurThreshold;
}