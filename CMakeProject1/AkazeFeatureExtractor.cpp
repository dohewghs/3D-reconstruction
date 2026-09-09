#include "AkazeFeatureExtractor.hpp"
AkazeFeatureExtractor::AkazeFeatureExtractor(float threshold, 
	cv::AKAZE::DescriptorType descriptorType, 
	int descriptorSize, 
	int descriptorChannels, 
	int nOctaves, 
	int nOctaveLayers)
{
	this->detector = cv::AKAZE::create(descriptorType, 
		descriptorSize, 
		descriptorChannels, 
		threshold, 
		nOctaves, 
		nOctaveLayers);
}

void AkazeFeatureExtractor::extract(Frame& frame)
{
	if (frame.imageGray.empty())
	{
		return;
	}

	frame.keypoints.clear();
	frame.descriptors.release();

	(*this->detector).detectAndCompute(
		frame.imageGray,
		cv::noArray(),
		frame.keypoints,
		frame.descriptors
	);
}

int AkazeFeatureExtractor::getDefaultNorm() const
{
	return (*this->detector).defaultNorm();
}

double AkazeFeatureExtractor::getThreshold() const
{
	return (*this->detector).getThreshold();
}
