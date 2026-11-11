#pragma once
#include "IFrameProvider.hpp"
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <iostream>

namespace fs = std::filesystem;


class DirectoryFrameProvider : public IFrameProvider
{
private:
	std::vector<fs::path> imagePaths;
	long currentFrameIndex;
	int maxDimension; // > 0: кадри зменшуються так, щоб більша сторона була <= maxDimension
public:
	DirectoryFrameProvider() = delete;
	explicit DirectoryFrameProvider(const std::string& dirPath, int maxDimension = 0);
	~DirectoryFrameProvider() = default;

	virtual bool getNextFrame(cv::Mat& outFrame) override;
	virtual bool hasNextFrame() const override;
	virtual void reset() override;

	size_t totalFrames();
	std::string currentFrameName() const; // ім'я файлу останнього виданого кадру
};