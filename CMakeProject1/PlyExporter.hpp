#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <concepts>
#include <opencv2/core.hpp>

#include "Types.hpp"

class PlyExporter {
private:
    // Перевірка на валідність та граничні межі координат
    static inline bool isValidPoint(const cv::Point3d& pt, double maxCoord = 100.0) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            return false;
        }
        return (std::abs(pt.x) <= maxCoord && 
                std::abs(pt.y) <= maxCoord && 
                std::abs(pt.z) <= maxCoord);
    }

    // Універсальний внутрішній метод збереження
    template <typename T, typename Extractor>
    static bool saveInternal(const std::string& filename, 
                             const std::vector<T>& items, 
                             Extractor extractPos, 
                             double maxCoord) 
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "[PlyExporter] Не вдалося відкрити файл для запису: " << filename << "\n";
            return false;
        }

        // Попередній відбір валідних точок для точного лічильника в заголовку
        std::vector<cv::Point3d> filtered;
        filtered.reserve(items.size());

        for (const auto& item : items) {
            cv::Point3d pt = extractPos(item);
            if (isValidPoint(pt, maxCoord)) {
                filtered.push_back(pt);
            }
        }

        // Запис PLY ASCII заголовка
        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << filtered.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "end_header\n";

        // Запис геометрії
        for (const auto& pt : filtered) {
            ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
        }

        ofs.close();
        return true;
    }

public:
    /// @brief Збереження хмари cv::Point3d
    /// @param maxCoord Порогове значення для фільтрації триангуляційних викидів (|x|, |y|, |z| <= maxCoord)
    static bool save(const std::string& filename, 
                     const std::vector<cv::Point3d>& points, 
                     double maxCoord = 100.0) 
    {
        return saveInternal(filename, points, [](const cv::Point3d& p) { return p; }, maxCoord);
    }

    /// @brief Збереження хмари MapPoint3D
    /// @param maxCoord Порогове значення для фільтрації триангуляційних викидів (|x|, |y|, |z| <= maxCoord)
    static bool save(const std::string& filename, 
                     const std::vector<MapPoint3D>& points, 
                     double maxCoord = 100.0) 
    {
        return saveInternal(filename, points, [](const MapPoint3D& mp) { return mp.position; }, maxCoord);
    }

    /// @brief Збереження щільної хмари з RGB-кольорами
    static bool saveColored(const std::string& filename, 
                            const std::vector<DensePoint3D>& points,
                            double maxCoord = 1e6) // масштаб монокулярного SfM довільний, тому 100 обрізало фасад
    {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return false;

        std::vector<DensePoint3D> filtered;
        filtered.reserve(points.size());

        for (const auto& p : points) {
            if (isValidPoint(p.position, maxCoord)) {
                filtered.push_back(p);
            }
        }

        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << filtered.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "property uchar red\n";
        ofs << "property uchar green\n";
        ofs << "property uchar blue\n";
        ofs << "end_header\n";

        for (const auto& p : filtered) {
            // OpenCV зберігає колір як BGR, тому записуємо: R = color[2], G = color[1], B = color[0]
            ofs << p.position.x << " " << p.position.y << " " << p.position.z << " "
                << static_cast<int>(p.color[2]) << " "
                << static_cast<int>(p.color[1]) << " "
                << static_cast<int>(p.color[0]) << "\n";
        }

        ofs.close();
        std::cout << "[PlyExporter] Збережено кольорову щільну хмару: " << filtered.size() 
                  << " точок у " << filename << std::endl;
        return true;
    }
};