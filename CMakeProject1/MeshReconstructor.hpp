#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "Types.hpp"

struct TriangleFace {
    int v0, v1, v2;
};

struct Mesh {
    std::vector<cv::Point3d> vertices;
    std::vector<TriangleFace> faces;
};

class MeshReconstructor {
private:
    double m_maxEdgeLength; // Максимальна довжина ребра між точками
    double m_maxCoord;      // Граничний поріг фільтрації викидів (|x|, |y|, |z| <= maxCoord)

    // Перевірка на валідність координат за аналогією з PlyExporter
    static inline bool isValidPoint(const cv::Point3d& pt, double maxCoord) {
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
            return false;
        }
        return (std::abs(pt.x) <= maxCoord && 
                std::abs(pt.y) <= maxCoord && 
                std::abs(pt.z) <= maxCoord);
    }

public:
    explicit MeshReconstructor(double maxEdgeLength = 2.5, double maxCoord = 100.0)
        : m_maxEdgeLength(maxEdgeLength), m_maxCoord(maxCoord) {}

    /**
     * Будує полігональну сітку методом 2.5D проекції Делоне з фільтрацією викидів
     */
    Mesh reconstructPlanarDelaunay(const std::vector<cv::Point3d>& rawPoints) {
        Mesh mesh;

        // 1. Фільтрація викидів та нескінченностей
        std::vector<cv::Point3d> validPoints;
        validPoints.reserve(rawPoints.size());

        for (const auto& pt : rawPoints) {
            if (isValidPoint(pt, m_maxCoord)) {
                validPoints.push_back(pt);
            }
        }

        if (validPoints.size() < 3) {
            std::cerr << "[Mesh] Недостатньо валідних точок для тріангуляції (" 
                      << validPoints.size() << " точок після фільтрації)." << std::endl;
            return mesh;
        }

        mesh.vertices = validPoints;

        // 2. Обчислення Bounding Box на площині XY
        double minX = 1e9, maxX = -1e9;
        double minY = 1e9, maxY = -1e9;

        for (const auto& pt : mesh.vertices) {
            minX = std::min(minX, pt.x);
            maxX = std::max(maxX, pt.x);
            minY = std::min(minY, pt.y);
            maxY = std::max(maxY, pt.y);
        }

        double padX = (maxX - minX) * 0.1 + 1.0;
        double padY = (maxY - minY) * 0.1 + 1.0;
        cv::Rect2d bounds(minX - padX, minY - padY, (maxX - minX) + 2 * padX, (maxY - minY) + 2 * padY);

        cv::Subdiv2D subdiv(bounds);

        // 3. Вставка точок у 2D Subdiv
        std::vector<cv::Point2f> points2D;
        points2D.reserve(mesh.vertices.size());

        for (const auto& pt : mesh.vertices) {
            cv::Point2f p2d(static_cast<float>(pt.x), static_cast<float>(pt.y));
            points2D.push_back(p2d);
            subdiv.insert(p2d);
        }

        // 4. Отримання трикутників Делоне
        std::vector<cv::Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);

        auto findClosestVertex = [&](const cv::Point2f& p) -> int {
            int bestIdx = -1;
            float minSqDist = 1e-4f;
            for (size_t i = 0; i < points2D.size(); ++i) {
                float dx = points2D[i].x - p.x;
                float dy = points2D[i].y - p.y;
                float sqDist = dx * dx + dy * dy;
                if (sqDist < minSqDist) {
                    minSqDist = sqDist;
                    bestIdx = static_cast<int>(i);
                }
            }
            return bestIdx;
        };

        // 5. Формування граней та відсікання занадто довгих ребер
        for (const auto& t : triangleList) {
            cv::Point2f ptA(t[0], t[1]);
            cv::Point2f ptB(t[2], t[3]);
            cv::Point2f ptC(t[4], t[5]);

            int idxA = findClosestVertex(ptA);
            int idxB = findClosestVertex(ptB);
            int idxC = findClosestVertex(ptC);

            if (idxA == -1 || idxB == -1 || idxC == -1) continue;
            if (idxA == idxB || idxB == idxC || idxA == idxC) continue;

            const auto& vA = mesh.vertices[idxA];
            const auto& vB = mesh.vertices[idxB];
            const auto& vC = mesh.vertices[idxC];

            double dAB = cv::norm(vA - vB);
            double dBC = cv::norm(vB - vC);
            double dCA = cv::norm(vC - vA);

            if (dAB > m_maxEdgeLength || dBC > m_maxEdgeLength || dCA > m_maxEdgeLength) {
                continue;
            }

            if (std::abs(vA.z - vB.z) > 1.5 || std::abs(vB.z - vC.z) > 1.5) {
                continue; // Не створювати грань між переднім планом і фасадом!
            }

            mesh.faces.push_back({idxA, idxB, idxC});
        }

        std::cout << "[Mesh] Побудовано граней (faces): " << mesh.faces.size() 
                  << " для " << mesh.vertices.size() << " вершин (після фільтрації)." << std::endl;
        return mesh;
    }

    /**
     * Збереження сітки у форматі .ply з підтримкою списку полігонів
     */
    static bool savePlyMesh(const std::string& filename, const Mesh& mesh) {
        std::ofstream ofs(filename);
        if (!ofs.is_open()) {
            std::cerr << "[Mesh] Не вдалося відкрити файл для запису: " << filename << std::endl;
            return false;
        }

        ofs << "ply\n";
        ofs << "format ascii 1.0\n";
        ofs << "element vertex " << mesh.vertices.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "element face " << mesh.faces.size() << "\n";
        ofs << "property list uchar int vertex_indices\n";
        ofs << "end_header\n";

        for (const auto& v : mesh.vertices) {
            ofs << v.x << " " << v.y << " " << v.z << "\n";
        }

        for (const auto& f : mesh.faces) {
            ofs << "3 " << f.v0 << " " << f.v1 << " " << f.v2 << "\n";
        }

        ofs.close();
        return true;
    }
};