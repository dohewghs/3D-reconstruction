#pragma once
#include "Types.hpp"

class IFeatureExtractor
{
public:
	virtual ~IFeatureExtractor() = default;

	virtual void extract(Frame& frame) = 0;
	virtual int getDefaultNorm() const = 0;
};

// AkazeExtractor
// SiftExtractor