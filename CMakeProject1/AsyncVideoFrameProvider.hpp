#pragma once


#include "IFrameProvider.hpp"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <queue>
#include <thread>

class AsyncVideoFrameProvider : IFrameProvider {
private:
    cv::VideoCapture cap;
    size_t frameStep;
    size_t maxBufferSize;

    std::queue<cv::Mat> buffer;
    std::mutex mtx;
    std::condition_variable cvNotEmpty;
    std::condition_variable cvNotFull;

    std::thread worker;
    std::atomic<bool> started{false};
    std::atomic<bool> stopRequested{false};
    bool eof = false;

    void markEof() {
        std::lock_guard<std::mutex> lock(mtx);
        eof = true;
        cvNotEmpty.notify_all();
    }

    void decodeLoop() {
        bool first = true;
        cv::Mat frame;

        while (!stopRequested) {
            if (!first) {
                for (int i = 0; i < frameStep - 1; ++i) {
                    if (!cap.grab()) {
                        markEof();
                        return;
                    }
                }
            }

            first = false;

            if (!cap.grab() || !cap.retrieve(frame))
            {
                markEof();
                return;
            }

            std::unique_lock<std::mutex> lock(mtx);
            cvNotFull.wait(lock, [this](){
                return buffer.size() < maxBufferSize || stopRequested;
            });

            if (stopRequested)
                return;

            buffer.push(std::move(frame));
            lock.unlock();

            cvNotEmpty.notify_one();
        }
    }

    void start() {
        if (!started.exchange(true)) {
            worker = std::thread(&AsyncVideoFrameProvider::decodeLoop, this);
        }
    }

    void stop() {
        stopRequested = true;
        cvNotEmpty.notify_all();
        cvNotFull.notify_all();

        if (worker.joinable()) {
            worker.join();
        }
    }
public:
    AsyncVideoFrameProvider(const std::string& path, size_t frameStep = 1, size_t maxBufferSize = 5)
        : cap(path), frameStep(frameStep), maxBufferSize(maxBufferSize) {
        if (!cap.isOpened()) {
            throw std::runtime_error("Failed to open video: " + path);
        }
    }

    ~AsyncVideoFrameProvider() {
        stop();
    }

    AsyncVideoFrameProvider(const AsyncVideoFrameProvider&) = delete;
    AsyncVideoFrameProvider& operator=(const AsyncVideoFrameProvider&) = delete;

    

    virtual bool getNextFrame(cv::Mat& outFrame) override {
        if (!started)
            start();

        std::unique_lock<std::mutex> lock(mtx);
        cvNotEmpty.wait(lock, [this](){
            return !buffer.empty() || eof || stopRequested;
        });

        if (buffer.empty())
            return false;

        outFrame = std::move(buffer.front());
        buffer.pop();

        lock.unlock();
        cvNotFull.notify_one();
        return true;
    }

    virtual void reset() override
    {
        // return;
    }
};