#pragma once

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <opencv2/core.hpp>

#include "Types.hpp"

class GlobalMap {
private:
    int nextPointId = 0;
    std::unordered_map<int, MapPoint3D> mapPoints; // <point_id, MapPoint3D>
    std::unordered_map<int, CameraPose> cameraPoses; // <frame_id, CameraPose>

    std::unordered_map<int, std::unordered_map<int, int>> frameToPointMap; // <frame_id, <keypoint_idx, point_id>>
    std::unordered_map<int, int> trackToPointMap; // <track_id, point_id>
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

    std::vector<int> getRegisteredFrameIds() const {
        std::vector<int> ids;
        ids.reserve(cameraPoses.size());
        for (const auto& [frameId, pose] : cameraPoses) ids.push_back(frameId);
        std::sort(ids.begin(), ids.end());
        return ids;
    }

    #pragma endregion

    #pragma region Map Points
    int addPoint3D(const cv::Point3d& position, const cv::Mat& descriptor,
               int frameId1, int keypointIdx1, int frameId2, int keypointIdx2)
    {
        return addPoint3D(position, descriptor, {{frameId1, keypointIdx1}, {frameId2, keypointIdx2}});
    }

    /**
     * Додає точку з довільною кількістю спостережень (тріангульований трек)
     */
    int addPoint3D(const cv::Point3d& position, const cv::Mat& descriptor,
                   const std::unordered_map<int, int>& observations, int trackId = -1)
    {
        MapPoint3D point;
        point.id = nextPointId++;
        point.trackId = trackId;
        point.position = position;
        point.descriptor = descriptor.clone();
        point.observations = observations;

        mapPoints[point.id] = point;

        // ВАЖЛИВО: Обов'язково оновлюємо зворотні індекси для всіх кадрів
        for (const auto& [frameId, keypointIdx] : observations) {
            frameToPointMap[frameId][keypointIdx] = point.id;
        }
        if (trackId >= 0) trackToPointMap[trackId] = point.id;

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

    void removeObservation(int pointId, int frameId) {
        auto it = mapPoints.find(pointId);
        if (it == mapPoints.end()) return;

        auto obsIt = it->second.observations.find(frameId);
        if (obsIt == it->second.observations.end()) return;

        auto frameIt = frameToPointMap.find(frameId);
        if (frameIt != frameToPointMap.end()) frameIt->second.erase(obsIt->second);
        it->second.observations.erase(obsIt);
    }

    void removePoint(int pointId) {
        auto it = mapPoints.find(pointId);
        if (it == mapPoints.end()) return;

        for (const auto& [frameId, keypointIdx] : it->second.observations) {
            auto frameIt = frameToPointMap.find(frameId);
            if (frameIt != frameToPointMap.end()) frameIt->second.erase(keypointIdx);
        }
        if (it->second.trackId >= 0) trackToPointMap.erase(it->second.trackId);
        mapPoints.erase(it);
    }

    void setPointPosition(int pointId, const cv::Point3d& position) {
        auto it = mapPoints.find(pointId);
        if (it != mapPoints.end()) it->second.position = position;
    }

    int getPointIdForFrameKeypoint(int frameId, int keypointIdx) const {
        auto frameIt = frameToPointMap.find(frameId);
        if (frameIt == frameToPointMap.end()) return -1; // Якщо кадр не знайдено

        auto keypointIt = frameIt->second.find(keypointIdx);
        if (keypointIt == frameIt->second.end()) return -1; // Якщо ключова точка не знайдена

        return keypointIt->second; // Повертаємо point_id
    }

    int getPointIdForTrack(int trackId) const {
        auto it = trackToPointMap.find(trackId);
        return (it == trackToPointMap.end()) ? -1 : it->second;
    }

    const MapPoint3D& getMapPoint(int pointId) const {
        auto it = mapPoints.find(pointId);
        if (it != mapPoints.end()) return it->second;
        throw std::runtime_error("Map point not found for ID: " + std::to_string(pointId));
    }

    void find2D3DCorrespondences(
        int refFrameId,
        const Frame& currentFrame,
        const std::vector<cv::DMatch>& matches,
        std::vector<cv::Point3f>& outPoints3D,
        std::vector<cv::Point2f>& outPoints2D,
        std::vector<int>& outPointIds,
        std::vector<int>& outCurrentKeypointIndices) const
    {
        outPoints3D.clear();
        outPoints2D.clear();
        outPointIds.clear();
        outCurrentKeypointIndices.clear();

        for (const auto& m : matches) {
            int refKeypointIdx = m.queryIdx;
            int currentKeypointIdx = m.trainIdx;

            int pointId = getPointIdForFrameKeypoint(refFrameId, refKeypointIdx);
            if (pointId == -1) continue;

            const auto& mapPointIt = mapPoints.find(pointId);
            if (mapPointIt == mapPoints.end()) continue;

            const MapPoint3D& mapPoint = mapPointIt->second;

            outPoints3D.emplace_back(
                static_cast<float>(mapPoint.position.x),
                static_cast<float>(mapPoint.position.y),
                static_cast<float>(mapPoint.position.z)
            );
            outPoints2D.push_back(currentFrame.keypoints[currentKeypointIdx].pt);
            outPointIds.push_back(pointId);
            outCurrentKeypointIndices.push_back(currentKeypointIdx);
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

    std::vector<cv::Point3d> getAllPointPositions() const {
        std::vector<cv::Point3d> positions;
        positions.reserve(mapPoints.size());
        for (const auto& entry : mapPoints) {
            positions.push_back(entry.second.position);
        }
        return positions;
    }
    #pragma endregion
};
