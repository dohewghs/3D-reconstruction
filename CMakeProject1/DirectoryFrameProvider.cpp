#include "DirectoryFrameProvider.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

DirectoryFrameProvider::DirectoryFrameProvider(const std::string& _dirPath) : IFrameProvider()
{
	if (!fs::exists(_dirPath) || !fs::is_directory(_dirPath)) {
		std::cerr << "Помилка: директорія не існує: " << _dirPath << std::endl;
		return;
	}

	for (const auto& entry : fs::directory_iterator(_dirPath))
	{
		if (entry.is_regular_file())
		{
			std::string ext = entry.path().extension().string();

			std::transform(ext.begin(), ext.end(), ext.begin(), std::tolower);

			if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
			{
				this->imagePaths.push_back(entry.path());
			}
		}
	}

	std::sort(this->imagePaths.begin(), this->imagePaths.end());

	this->currentFrameIndex = 0;

	// можна додати логування
}

bool DirectoryFrameProvider::getNextFrame(cv::Mat& outFrame)
{
	while (this->hasNextFrame())
	{
		const fs::path& currentPath = this->imagePaths[this->currentFrameIndex++];

		outFrame = cv::imread(currentPath.string(), cv::IMREAD_COLOR);

		if (outFrame.empty())
		{
			//std::cerr << "Warning: file is broken! Path: " << currentPath << std::endl;
			continue;
		}

		return true;
	}

	return false;
}

bool DirectoryFrameProvider::hasNextFrame() const
{
	return this->currentFrameIndex < this->imagePaths.size();
}

void DirectoryFrameProvider::reset()
{
	this->currentFrameIndex = 0;
}

size_t DirectoryFrameProvider::totalFrames()
{
	return this->imagePaths.size();
}
