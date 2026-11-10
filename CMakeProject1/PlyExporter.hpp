#pragma once

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>

#include "Types.hpp"

class PlyExporter {
public:
    static bool save(const std::string& filename, const std::vector<cv::Point3d>& points) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "Не вдалося відкрити файл для запису: " << filename << std::endl;
            return false;
        }

        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << points.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "end_header\n";

        for (const auto& pt : points) {
            ofs << pt.x << " " << pt.y << " " << pt.z << "\n";
        }

        ofs.close();
        return true;
    }

    static bool save(const std::string& filename, const std::vector<MapPoint3D>& points) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) return false;

        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << points.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "end_header\n";

        for (const auto& pt : points) {
            ofs << pt.position.x << " " 
                << pt.position.y << " " 
                << pt.position.z << "\n";
        }
        return true;
    }
};