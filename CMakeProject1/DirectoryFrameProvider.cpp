#include "DirectoryFrameProvider.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

DirectoryFrameProvider::DirectoryFrameProvider(const std::string& _dirPath, int _maxDimension)
	: IFrameProvider(), currentFrameIndex(0), maxDimension(_maxDimension)
{
	if (!fs::exists(_dirPath) || !fs::is_directory(_dirPath)) {
		std::cerr << "Error: Directory isn't exist: " << _dirPath << std::endl;
		return;
	}

	for (const auto& entry : fs::directory_iterator(_dirPath))
	{
		if (entry.is_regular_file())
		{
			std::string ext = entry.path().extension().string();

			std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {return std::tolower(c); });

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

		// 12 Мп фото не потрібні для SfM: на ~2000 px ознаки стабільніші, а все працює в рази швидше
		int largest = std::max(outFrame.cols, outFrame.rows);
		if (this->maxDimension > 0 && largest > this->maxDimension)
		{
			double scale = static_cast<double>(this->maxDimension) / largest;
			cv::resize(outFrame, outFrame, cv::Size(), scale, scale, cv::INTER_AREA);
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

std::string DirectoryFrameProvider::currentFrameName() const
{
	if (this->currentFrameIndex <= 0 || this->currentFrameIndex > static_cast<long>(this->imagePaths.size()))
		return "";
	return this->imagePaths[this->currentFrameIndex - 1].filename().string();
}
