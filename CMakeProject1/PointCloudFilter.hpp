#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "Types.hpp"

class PointCloudFilter {
private:
    struct VoxelAccumulator {
        cv::Vec3d positionSum = cv::Vec3d(0, 0, 0);
        cv::Vec3d colorSum = cv::Vec3d(0, 0, 0);
        int count = 0;
        int distinctSources = 0;
        int lastSource = INT32_MIN;
    };

    static constexpr int64_t kAxisOffset = 1 << 20; // 21 біт на вісь

    static bool voxelKey(const cv::Point3d& p, double voxelSize, int64_t& outKey) {
        int64_t ix = static_cast<int64_t>(std::floor(p.x / voxelSize)) + kAxisOffset;
        int64_t iy = static_cast<int64_t>(std::floor(p.y / voxelSize)) + kAxisOffset;
        int64_t iz = static_cast<int64_t>(std::floor(p.z / voxelSize)) + kAxisOffset;

        const int64_t limit = 2 * kAxisOffset;
        if (ix < 0 || iy < 0 || iz < 0 || ix >= limit || iy >= limit || iz >= limit) return false;

        outKey = (ix << 42) | (iy << 21) | iz;
        return true;
    }

public:
    /**
     * Злиття щільної хмари у воксельну сітку.
     *
     * Кожна стерео-пара дає власний шар точок зі своєю похибкою, тож просте об'єднання пар
     * дає "багатошаровий", розмитий фасад з купою викидів. Тут:
     *  - точки в одному вокселі усереднюються (позиція та колір) -> один тонкий шар;
     *  - воксель залишається, лише якщо його підтвердили щонайменше minSources РІЗНИХ пар
     *    (геометрична узгодженість між видами) — випадкові хибні збіги SGBM так відсіюються.
     *
     * Вимога: точки однієї пари мають іти суцільним блоком (як їх і додає пайплайн).
     */
    static std::vector<DensePoint3D> voxelFuse(
        const std::vector<DensePoint3D>& points,
        double voxelSize,
        int minSources)
    {
        std::vector<DensePoint3D> fused;
        if (points.empty() || voxelSize <= 0.0) return fused;

        std::unordered_map<int64_t, VoxelAccumulator> voxels;
        voxels.reserve(points.size() / 2);

        for (const auto& p : points) {
            if (!std::isfinite(p.position.x) || !std::isfinite(p.position.y) || !std::isfinite(p.position.z)) continue;

            int64_t key;
            if (!voxelKey(p.position, voxelSize, key)) continue;

            VoxelAccumulator& acc = voxels[key];
            acc.positionSum += cv::Vec3d(p.position.x, p.position.y, p.position.z);
            acc.colorSum += cv::Vec3d(p.color[0], p.color[1], p.color[2]);
            acc.count++;

            if (p.sourceId != acc.lastSource) {
                acc.lastSource = p.sourceId;
                acc.distinctSources++;
            }
        }

        fused.reserve(voxels.size());
        for (const auto& entry : voxels) {
            const VoxelAccumulator& acc = entry.second;
            if (acc.distinctSources < minSources) continue;

            cv::Vec3d pos = acc.positionSum / acc.count;
            cv::Vec3d col = acc.colorSum / acc.count;

            DensePoint3D out;
            out.position = cv::Point3d(pos[0], pos[1], pos[2]);
            out.color = cv::Vec3b(cv::saturate_cast<uchar>(col[0]),
                                  cv::saturate_cast<uchar>(col[1]),
                                  cv::saturate_cast<uchar>(col[2]));
            out.sourceId = acc.distinctSources;
            fused.push_back(out);
        }

        std::cout << "[Фільтр] Воксель " << voxelSize << ": " << points.size() << " -> "
                  << voxels.size() << " вокселів -> " << fused.size()
                  << " точок (підтверджені >= " << minSources << " парами)." << std::endl;

        return fused;
    }
};
