#pragma once

#include <vector>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <opencv2/core.hpp>

#include "Types.hpp"
#include "GeometryUtils.hpp"

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
     * Злиття карт глибин з перевіркою узгодженості між видами.
     *
     * Кожен піксель карти i переводиться у світ і проєктується в опорні кадри інших карт.
     * Якщо там є глибина, що відрізняється не більше ніж на relTolerance, — це підтвердження
     * від незалежного виду. Точка лишається, якщо підтверджень >= minConsistent; її позиція
     * усереднюється з підтвердженнями. Хибні збіги SGBM не узгоджуються між різними парами
     * і відсіюються.
     *
     * @param rawOut         якщо не nullptr — сюди потрапляють усі точки до фільтрації (для порівняння)
     * @param fusedDepthOut  якщо не nullptr — для кожної карти: глибина узгоджених (усереднених) точок
     *                       у її опорному кадрі, 0 там, де точку відкинуто (потрібно для побудови поверхні)
     */
    static std::vector<DensePoint3D> fuseDepthMaps(
        const std::vector<DepthMap>& maps,
        const std::unordered_map<int, CameraPose>& poses,
        const cv::Mat& cameraMatrix,
        const std::unordered_map<int, cv::Mat>& colorImages,
        int minConsistent = 1,
        double relTolerance = 0.015,
        std::vector<DensePoint3D>* rawOut = nullptr,
        std::vector<cv::Mat>* fusedDepthOut = nullptr)
    {
        std::vector<DensePoint3D> fused;
        cv::Mat Kd;
        cameraMatrix.convertTo(Kd, CV_64F);
        const double fx = Kd.at<double>(0, 0), fy = Kd.at<double>(1, 1);
        const double cx = Kd.at<double>(0, 2), cy = Kd.at<double>(1, 2);

        // Пози опорних кадрів карт — один раз, а не на кожен піксель
        std::vector<cv::Matx33d> rotations(maps.size());
        std::vector<cv::Vec3d> translations(maps.size());
        for (size_t i = 0; i < maps.size(); ++i) {
            rotations[i] = geom::rotation(poses.at(maps[i].refId));
            translations[i] = geom::translation(poses.at(maps[i].refId));
        }

        for (size_t i = 0; i < maps.size(); ++i) {
            const DepthMap& dm = maps[i];
            cv::Matx33d Rt = rotations[i].t();
            const cv::Vec3d& t = translations[i];
            const cv::Mat& color = colorImages.at(dm.refId);
            cv::Mat fusedDepth;
            if (fusedDepthOut) fusedDepth = cv::Mat::zeros(dm.depth.size(), CV_32F);

            // Інші карти, які можуть підтвердити точку (не з того ж кадру і не з тієї ж пари)
            std::vector<size_t> others;
            for (size_t j = 0; j < maps.size(); ++j) {
                if (j == i || maps[j].refId == dm.refId) continue;
                bool samePair = (maps[j].refId == dm.srcId && maps[j].srcId == dm.refId);
                if (!samePair) others.push_back(j);
            }

            size_t kept = 0, total = 0;
            for (int v = 0; v < dm.depth.rows; ++v) {
                const float* row = dm.depth.ptr<float>(v);
                for (int u = 0; u < dm.depth.cols; ++u) {
                    double z = row[u];
                    if (z <= 0) continue;
                    total++;

                    cv::Vec3d Xc((u - cx) / fx * z, (v - cy) / fy * z, z);
                    cv::Vec3d Xw = Rt * (Xc - t);

                    cv::Vec3d sum = Xw;
                    int support = 0;
                    for (size_t j : others) {
                        const DepthMap& other = maps[j];
                        cv::Vec3d Xo = rotations[j] * Xw + translations[j];
                        if (!(Xo[2] > 0)) continue;
                        int uo = cvRound(fx * Xo[0] / Xo[2] + cx);
                        int vo = cvRound(fy * Xo[1] / Xo[2] + cy);
                        if (uo < 0 || vo < 0 || uo >= other.depth.cols || vo >= other.depth.rows) continue;
                        double zo = other.depth.at<float>(vo, uo);
                        if (zo <= 0 || std::abs(zo - Xo[2]) > relTolerance * zo) continue;

                        cv::Vec3d XoMeasured((uo - cx) / fx * zo, (vo - cy) / fy * zo, zo);
                        sum += rotations[j].t() * (XoMeasured - translations[j]);
                        support++;
                    }

                    DensePoint3D p;
                    p.color = color.at<cv::Vec3b>(v, u);
                    p.sourceId = static_cast<int>(i);
                    if (rawOut) {
                        p.position = geom::toPoint(Xw);
                        rawOut->push_back(p);
                    }
                    if (support >= minConsistent) {
                        cv::Vec3d mean = sum / static_cast<double>(support + 1);
                        p.position = geom::toPoint(mean);
                        fused.push_back(p);
                        if (fusedDepthOut) {
                            double zf = (rotations[i] * mean + t)[2];
                            if (zf > 0) fusedDepth.at<float>(v, u) = static_cast<float>(zf);
                        }
                        kept++;
                    }
                }
            }

            if (fusedDepthOut) fusedDepthOut->push_back(fusedDepth);
            std::cout << "[Злиття] Карта #" << dm.refId << " (пара з #" << dm.srcId << "): "
                      << total << " пікселів, узгоджених: " << kept << std::endl;
        }
        return fused;
    }

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
