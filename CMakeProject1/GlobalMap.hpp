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

    std::unordered_map<int, std::unordered_map<int, int>> frameToPointMap; // <frame_id, <keypoint_idx, point_id>>
public:
    GlobalMap() = default;

    #pragma region Camera Poses
    void addCameraPose(int frameId, const CameraPose& pose) {
        cameraPoses[frameId] = pose;
    }

    bool hasCameraPose(int frameId) const {
        return cameraPoses.find(frameId) != cameraPoses.end();
    }

    const CameraPose& getCameraPose(int frameId) const {
        auto it = cameraPoses.find(frameId);
        if (it != cameraPoses.end()) {
            return it->second;
        }
        throw std::runtime_error("Camera pose not found for frame ID: " + std::to_string(frameId));
    }

    #pragma endregion

    #pragma region Map Points
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

    void addObservation(int pointId, int frameId, int keypointIdx) {
        auto it = mapPoints.find(pointId);

        if (it != mapPoints.end()) {
            it->second.observations[frameId] = keypointIdx;
            this->frameToPointMap[frameId][keypointIdx] = pointId;
        } else {
            throw std::runtime_error("Map point not found for ID: " + std::to_string(pointId));
        }
    }

    int getPointIdForFrameKeypoint(int frameId, int keypointIdx) const {
        auto frameIt = frameToPointMap.find(frameId);
        if (frameIt == frameToPointMap.end()) return -1; // Якщо кадр не знайдено
        
        auto keypointIt = frameIt->second.find(keypointIdx);
        if (keypointIt == frameIt->second.end()) return -1; // Якщо ключова точка не знайдена
            
        return keypointIt->second; // Повертаємо point_id
    }


    void find2D3DCorrespondences(
        int refFrameId,
        const Frame& currentFrame,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Point3f>& outPoints3D,
        std::vector<cv::Point2f>& outPoints2D,
        std::vector<int>& outPointIds) const {
        outPoints3D.clear();
        outPoints2D.clear();
        outPointIds.clear();

        for (const auto& m : matches) {
            int refKeypointIdx = m.queryIdx;
            int currentKeypointIdx = m.trainIdx;

            int pointId = getPointIdForFrameKeypoint(refFrameId, refKeypointIdx);

            if (pointId == -1) continue; // Якщо немає відповідної 3D-точки, пропускаємо

            const auto& mapPointIt = mapPoints.find(pointId);
            if (mapPointIt == mapPoints.end()) continue; // Якщо 3D-точка не знайдена, пропускаємо

            const MapPoint3D& mapPoint = mapPointIt->second;

            outPoints3D.push_back(cv::Point3f(
                static_cast<float>(mapPoint.position.x),
                static_cast<float>(mapPoint.position.y),
                static_cast<float>(mapPoint.position.z)
            ));

            outPoints2D.push_back(currentFrame.keypoints[currentKeypointIdx].pt);
            outPointIds.push_back(pointId);
        }
    }
    #pragma endregion

    #pragma region Getters
    std::vector<MapPoint3D> getAllMapPoints() const {
        std::vector<MapPoint3D> points;
        for (const auto& entry : mapPoints) {
            points.push_back(entry.second);
        }
        return points;
    }

    size_t getMapPointCount() const {
        return mapPoints.size();
    }

    size_t getCameraPoseCount() const {
        return cameraPoses.size();
    }

    const std::unordered_map<int, CameraPose>& getCameraPoses() const {
        return cameraPoses;
    }

    const std::unordered_map<int, MapPoint3D>& getMapPoints() const {
        return mapPoints;
    }
    #pragma endregion
};