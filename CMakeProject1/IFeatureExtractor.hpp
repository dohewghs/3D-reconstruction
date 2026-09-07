#pragma once
#include "Types.hpp"

class IFeatureExtractor
{
	virtual ~IFeatureExtractor() = default;

	virtual void extract(Frame& frame) = 0;
};

// AkazeExtractor
// SiftExtractor