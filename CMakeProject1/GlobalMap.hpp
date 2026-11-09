#pragma once

#include <unordered_map>
#include <vector>
#include <opencv2/core.hpp>

#include "Types.hpp"

class GlobalMap {
private:
    int nextPointId = 0;
    std::unordered_map<int, MapPoint3D> mapPoints; // <point_id, MapPoint3D>
    std::unordered_map<int, CameraPose> cameraPoses; // <frame_id, CameraPose>

public:
    void addCameraPose(int frameId, const CameraPose& pose) {
        cameraPoses[frameId] = pose;
    }

    int addPoint3D(const cv::Point3d& position, const cv::Mat& descriptor, int frameId1, int keypointIdx1, int frameId2, int keypointIdx2) {
        MapPoint3D point;
        point.id = nextPointId++;
        point.position = position;
        point.descriptor = descriptor.clone(); // Копіюємо дескриптор
        point.observations[frameId1] = keypointIdx1;
        point.observations[frameId2] = keypointIdx2;

        mapPoints[point.id] = point;
        return point.id;
    }

    std::vector<MapPoint3D> getAllMapPoints() const {
        std::vector<MapPoint3D> points;
        for (const auto& entry : mapPoints) {
            points.push_back(entry.second);
        }
        return points;
    }

    const std::unordered_map<int, CameraPose>& getCameraPoses() const {
        return cameraPoses;
    }

    const std::unordered_map<int, MapPoint3D>& getMapPoints() const {
        return mapPoints;
    }
};