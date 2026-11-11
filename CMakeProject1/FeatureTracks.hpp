#pragma once

#include <vector>
#include <map>
#include <algorithm>
#include <unordered_map>
#include <numeric>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

#include "Types.hpp"
#include "IFeatureMatcher.hpp"

/**
 * Трек — одна фізична точка сцени, яку бачать кілька кадрів: <frame_id, keypoint_idx>.
 *
 * Чому не лише сусідні кадри: точку, яку бачать кадри 6 і 8, але не бачить 7, послідовний трекінг
 * втрачає, і PnP для кадру 8 залишається без 2D-3D відповідностей. Треки поєднують збіги з УСІХ пар.
 */
struct Track {
    std::unordered_map<int, int> observations; // <frame_id, keypoint_idx>
};

struct FeatureTracks {
    std::vector<Track> tracks;
    std::unordered_map<int, std::vector<int>> keypointToTrack;    // frame_id -> (keypoint_idx -> track_id або -1)
    std::map<std::pair<int, int>, std::vector<cv::DMatch>> pairMatches; // (a < b) -> геометрично перевірені збіги

    int trackOf(int frameId, int keypointIdx) const {
        auto it = keypointToTrack.find(frameId);
        if (it == keypointToTrack.end() || keypointIdx < 0 || keypointIdx >= static_cast<int>(it->second.size())) return -1;
        return it->second[keypointIdx];
    }

    // Збіги a -> b (queryIdx належить a). Порожньо, якщо пара не пройшла перевірку.
    std::vector<cv::DMatch> matchesBetween(int a, int b) const {
        if (a < b) {
            auto it = pairMatches.find({a, b});
            return (it == pairMatches.end()) ? std::vector<cv::DMatch>{} : it->second;
        }
        auto it = pairMatches.find({b, a});
        if (it == pairMatches.end()) return {};
        std::vector<cv::DMatch> swapped;
        swapped.reserve(it->second.size());
        for (const auto& m : it->second) swapped.emplace_back(m.trainIdx, m.queryIdx, m.distance);
        return swapped;
    }

    size_t matchCount(int a, int b) const {
        auto it = pairMatches.find({std::min(a, b), std::max(a, b)});
        return (it == pairMatches.end()) ? 0 : it->second.size();
    }
};

class TrackBuilder {
private:
    int m_matchWindow;          // Зіставляємо кадри, що відстоять не більше ніж на N позицій (0 = всі пари)
    int m_minPairInliers;       // Мінімум збігів, які пройшли перевірку фундаментальною матрицею
    double m_ransacThresholdPx;

    class UnionFind {
        std::vector<int> parent;
    public:
        explicit UnionFind(size_t n) : parent(n) { std::iota(parent.begin(), parent.end(), 0); }
        int find(int x) {
            while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
            return x;
        }
        void unite(int a, int b) { a = find(a); b = find(b); if (a != b) parent[a] = b; }
    };

public:
    explicit TrackBuilder(int matchWindow = 10, int minPairInliers = 30, double ransacThresholdPx = 2.0)
        : m_matchWindow(matchWindow), m_minPairInliers(minPairInliers), m_ransacThresholdPx(ransacThresholdPx) {}

    FeatureTracks build(const std::vector<int>& frameIds,
                        const std::unordered_map<int, Frame>& frames,
                        IFeatureMatcher& matcher) const
    {
        FeatureTracks result;

        // 1. Зіставлення пар + геометрична верифікація (RANSAC по фундаментальній матриці)
        for (size_t i = 0; i < frameIds.size(); ++i) {
            for (size_t j = i + 1; j < frameIds.size(); ++j) {
                if (m_matchWindow > 0 && static_cast<int>(j - i) > m_matchWindow) break;

                const Frame& fa = frames.at(frameIds[i]);
                const Frame& fb = frames.at(frameIds[j]);
                if (fa.descriptors.empty() || fb.descriptors.empty()) continue;

                auto matches = matcher.match(fa.descriptors, fb.descriptors);
                if (static_cast<int>(matches.size()) < m_minPairInliers) continue;

                std::vector<cv::Point2f> pa, pb;
                pa.reserve(matches.size());
                pb.reserve(matches.size());
                for (const auto& m : matches) {
                    pa.push_back(fa.keypoints[m.queryIdx].pt);
                    pb.push_back(fb.keypoints[m.trainIdx].pt);
                }

                cv::Mat mask;
                cv::Mat F = cv::findFundamentalMat(pa, pb, cv::FM_RANSAC, m_ransacThresholdPx, 0.999, mask);
                if (F.empty() || mask.empty()) continue;

                std::vector<cv::DMatch> inliers;
                for (size_t k = 0; k < matches.size(); ++k) {
                    if (mask.at<uchar>(static_cast<int>(k))) inliers.push_back(matches[k]);
                }

                if (static_cast<int>(inliers.size()) >= m_minPairInliers) {
                    result.pairMatches[{frameIds[i], frameIds[j]}] = std::move(inliers);
                }
            }
        }

        std::cout << "[Треки] Перевірених пар кадрів: " << result.pairMatches.size() << std::endl;
        for (const auto& [pair, matches] : result.pairMatches) {
            std::cout << "   #" << pair.first << " <-> #" << pair.second << ": " << matches.size() << " збігів" << std::endl;
        }

        // 2. Об'єднання збігів у треки (union-find над усіма ключовими точками)
        std::unordered_map<int, int> offset;
        int total = 0;
        for (int id : frameIds) {
            offset[id] = total;
            total += static_cast<int>(frames.at(id).keypoints.size());
        }

        UnionFind uf(total);
        for (const auto& [pair, matches] : result.pairMatches) {
            for (const auto& m : matches) {
                uf.unite(offset[pair.first] + m.queryIdx, offset[pair.second] + m.trainIdx);
            }
        }

        std::unordered_map<int, Track> groups;
        std::unordered_map<int, bool> conflicted;
        for (const auto& [pair, matches] : result.pairMatches) {
            for (const auto& m : matches) {
                for (auto [frameId, kpIdx] : {std::pair{pair.first, m.queryIdx}, std::pair{pair.second, m.trainIdx}}) {
                    int root = uf.find(offset[frameId] + kpIdx);
                    auto& obs = groups[root].observations;
                    auto it = obs.find(frameId);
                    if (it != obs.end() && it->second != kpIdx) conflicted[root] = true; // дві різні точки одного кадру
                    obs[frameId] = kpIdx;
                }
            }
        }

        for (int id : frameIds) {
            result.keypointToTrack[id].assign(frames.at(id).keypoints.size(), -1);
        }

        int longTracks = 0;
        for (auto& [root, track] : groups) {
            if (conflicted.count(root)) continue;
            int trackId = static_cast<int>(result.tracks.size());
            for (const auto& [frameId, kpIdx] : track.observations) {
                result.keypointToTrack[frameId][kpIdx] = trackId;
            }
            if (track.observations.size() >= 3) longTracks++;
            result.tracks.push_back(std::move(track));
        }

        std::cout << "[Треки] Побудовано треків: " << result.tracks.size()
                  << " (з них у 3+ кадрах: " << longTracks << ")" << std::endl;
        return result;
    }
};
