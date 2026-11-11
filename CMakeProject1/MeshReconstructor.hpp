#pragma once

#include <vector>
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "Types.hpp"
#include "GeometryUtils.hpp"

struct ColoredMesh {
    std::vector<cv::Vec3f> vertices;
    std::vector<cv::Vec3b> colors;   // BGR
    std::vector<cv::Vec3i> faces;
};

/**
 * Суцільна поверхня з карт глибин (depth map meshing).
 *
 * Кожна карта глибин — це регулярна сітка пікселів, тож сусідні пікселі природно з'єднуються
 * у трикутники. Трикутник відкидається, якщо він перетинає розрив глибини (край будинку,
 * перехід фасад/небо) або видимий майже "по дотичній" (розтягнуті смуги на силуетах).
 *
 * Перед тріангуляцією:
 *  - дрібні "острівці" шуму видаляються;
 *  - невеликі дірки (скло, однотонні ділянки стіни) заповнюються інтерполяцією ОБЕРНЕНОЇ глибини:
 *    для площини 1/z лінійна в координатах зображення, тож плоский фасад заповнюється точно;
 *  - ділянки, які вже покриті попередньою картою, пропускаються — інакше шари з різних карт
 *    накладаються один на одного.
 * Колір вершини береться з відповідного пікселя фото.
 */
class MeshReconstructor {
private:
    int m_step;                 // Крок сітки у пікселях
    double m_maxDepthJump;      // Максимальна відносна різниця глибин у межах трикутника
    double m_minViewCos;        // Мінімальний |cos| між нормаллю трикутника і променем камери
    int m_maxHoleArea;          // Максимальна площа дірки (пікселі), яку можна заповнити
    int m_minIslandArea;        // Менші острівці валідних пікселів вважаються шумом
    double m_overlapTolerance;  // Відносна розбіжність глибин, за якої ділянку вважаємо вже покритою
    int m_minComponentFaces;    // Відокремлені шматки сітки з меншою кількістю трикутників видаляються

    cv::Mat cleanAndFill(const cv::Mat& depth) const {
        cv::Mat valid = depth > 0;

        // 1. Острівці шуму
        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(valid, labels, stats, centroids, 8, CV_32S);
        cv::Mat cleaned = depth.clone();
        for (int y = 0; y < labels.rows; ++y) {
            const int* l = labels.ptr<int>(y);
            float* d = cleaned.ptr<float>(y);
            for (int x = 0; x < labels.cols; ++x) {
                if (l[x] > 0 && stats.at<int>(l[x], cv::CC_STAT_AREA) < m_minIslandArea) d[x] = 0.f;
            }
        }
        valid = cleaned > 0;

        // 2. Дірки: компоненти невалідних пікселів, що не торкаються краю кадру і не завеликі
        cv::Mat invalid = ~valid;
        n = cv::connectedComponentsWithStats(invalid, labels, stats, centroids, 4, CV_32S);
        std::vector<char> fillable(n, 0);
        for (int i = 1; i < n; ++i) {
            int x = stats.at<int>(i, cv::CC_STAT_LEFT), y = stats.at<int>(i, cv::CC_STAT_TOP);
            int w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            bool touchesBorder = x == 0 || y == 0 || x + w >= labels.cols || y + h >= labels.rows;
            fillable[i] = !touchesBorder && stats.at<int>(i, cv::CC_STAT_AREA) <= m_maxHoleArea;
        }

        // Межа дірки має лежати на одній поверхні: інакше це розрив (наприклад, між балконом і стіною)
        std::vector<float> minInv(n, 1e30f), maxInv(n, 0.f);
        for (int y = 1; y + 1 < labels.rows; ++y) {
            for (int x = 1; x + 1 < labels.cols; ++x) {
                float z = cleaned.at<float>(y, x);
                if (z <= 0) continue;
                for (auto [dx, dy] : {std::pair{1, 0}, std::pair{-1, 0}, std::pair{0, 1}, std::pair{0, -1}}) {
                    int l = labels.at<int>(y + dy, x + dx);
                    if (l > 0 && fillable[l]) {
                        minInv[l] = std::min(minInv[l], 1.f / z);
                        maxInv[l] = std::max(maxInv[l], 1.f / z);
                    }
                }
            }
        }
        cv::Mat fillMask = cv::Mat::zeros(depth.size(), CV_8U);
        for (int y = 0; y < labels.rows; ++y) {
            const int* l = labels.ptr<int>(y);
            uchar* m = fillMask.ptr<uchar>(y);
            for (int x = 0; x < labels.cols; ++x) {
                int c = l[x];
                if (c > 0 && fillable[c] && maxInv[c] > 0 && (maxInv[c] - minInv[c]) <= 0.1f * maxInv[c]) m[x] = 255;
            }
        }

        // 3. Заповнення: дифузія оберненої глибини від краю дірки всередину (кільце за кільцем)
        cv::Mat inv = cv::Mat::zeros(depth.size(), CV_32F), known;
        cv::divide(1.0, cleaned, inv);
        inv.setTo(0, ~valid);
        valid.convertTo(known, CV_32F, 1.0 / 255.0);

        for (int iter = 0; iter < 200; ++iter) {
            cv::Mat sumInv, sumKnown;
            cv::boxFilter(inv.mul(known), sumInv, CV_32F, cv::Size(3, 3), cv::Point(-1, -1), false);
            cv::boxFilter(known, sumKnown, CV_32F, cv::Size(3, 3), cv::Point(-1, -1), false);
            cv::Mat frontier = (sumKnown > 0) & (known == 0) & fillMask;
            if (cv::countNonZero(frontier) == 0) break;
            cv::Mat estimate = sumInv / (sumKnown + 1e-12);
            estimate.copyTo(inv, frontier);
            known.setTo(1.0, frontier);
        }

        cv::Mat filled = cv::Mat::zeros(depth.size(), CV_32F);
        for (int y = 0; y < filled.rows; ++y) {
            const float* i = inv.ptr<float>(y);
            const float* k = known.ptr<float>(y);
            float* f = filled.ptr<float>(y);
            for (int x = 0; x < filled.cols; ++x) {
                if (k[x] > 0 && i[x] > 0) f[x] = 1.f / i[x];
            }
        }
        return filled;
    }

public:
    explicit MeshReconstructor(int step = 2,
                               double maxDepthJump = 0.03,
                               double minViewCos = 0.1,
                               int maxHoleArea = 20000,
                               int minIslandArea = 150,
                               double overlapTolerance = 0.02,
                               int minComponentFaces = 300)
        : m_step(step), m_maxDepthJump(maxDepthJump), m_minViewCos(minViewCos),
          m_maxHoleArea(maxHoleArea), m_minIslandArea(minIslandArea), m_overlapTolerance(overlapTolerance),
          m_minComponentFaces(minComponentFaces) {}

    /**
     * @param depths       узгоджені глибини (з PointCloudFilter::fuseDepthMaps), по одній на карту
     * @param maps         відповідні карти (потрібен refId)
     * @param cameraMatrix K роздільності карт глибин
     * @param colorImages  кольорові кадри тієї ж роздільності
     */
    ColoredMesh buildFromDepthMaps(const std::vector<cv::Mat>& depths,
                                   const std::vector<DepthMap>& maps,
                                   const std::unordered_map<int, CameraPose>& poses,
                                   const cv::Mat& cameraMatrix,
                                   const std::unordered_map<int, cv::Mat>& colorImages) const
    {
        ColoredMesh mesh;
        cv::Mat Kd;
        cameraMatrix.convertTo(Kd, CV_64F);
        const double fx = Kd.at<double>(0, 0), fy = Kd.at<double>(1, 1);
        const double cx = Kd.at<double>(0, 2), cy = Kd.at<double>(1, 2);

        std::vector<cv::Mat> filled(depths.size());
        for (size_t i = 0; i < depths.size(); ++i) filled[i] = cleanAndFill(depths[i]);

        // Найповніші карти — першими: вони задають основну поверхню, решта лише доповнює
        std::vector<size_t> order(depths.size());
        std::iota(order.begin(), order.end(), 0);
        std::vector<int> validCount(depths.size());
        for (size_t i = 0; i < depths.size(); ++i) validCount[i] = cv::countNonZero(filled[i]);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return validCount[a] > validCount[b]; });

        // Глибина лише там, де карта справді дала трикутники: "покритими" вважаємо тільки ці ділянки,
        // інакше місця, де попередня карта трикутників не побудувала, не заповнить ніхто
        std::vector<cv::Mat> coverage(depths.size());
        std::vector<size_t> processed;
        for (size_t idx : order) {
            const cv::Mat& D = filled[idx];
            const CameraPose& pose = poses.at(maps[idx].refId);
            cv::Matx33d R = geom::rotation(pose), Rt = R.t();
            cv::Vec3d t = geom::translation(pose);
            const cv::Mat& color = colorImages.at(maps[idx].refId);

            int gridRows = (D.rows - 1) / m_step + 1, gridCols = (D.cols - 1) / m_step + 1;
            std::vector<int> vertexIndex(static_cast<size_t>(gridRows) * gridCols, -1);
            std::vector<cv::Vec3d> camPoints(vertexIndex.size());
            size_t firstVertex = mesh.vertices.size(), firstFace = mesh.faces.size();

            for (int r = 0; r < gridRows; ++r) {
                for (int c = 0; c < gridCols; ++c) {
                    int v = r * m_step, u = c * m_step;
                    double z = D.at<float>(v, u);
                    if (z <= 0) continue;

                    cv::Vec3d Xc((u - cx) / fx * z, (v - cy) / fy * z, z);
                    cv::Vec3d Xw = Rt * (Xc - t);

                    // Вже покрито поверхнею з попередньої карти?
                    bool covered = false;
                    for (size_t j : processed) {
                        const CameraPose& otherPose = poses.at(maps[j].refId);
                        cv::Vec3d Xo = geom::toCamera(otherPose, Xw);
                        if (!(Xo[2] > 0)) continue;
                        int uo = cvRound(fx * Xo[0] / Xo[2] + cx), vo = cvRound(fy * Xo[1] / Xo[2] + cy);
                        if (uo < 0 || vo < 0 || uo >= coverage[j].cols || vo >= coverage[j].rows) continue;
                        float zo = coverage[j].at<float>(vo, uo);
                        if (zo > 0 && std::abs(zo - Xo[2]) < m_overlapTolerance * zo) { covered = true; break; }
                    }
                    if (covered) continue;

                    size_t g = static_cast<size_t>(r) * gridCols + c;
                    vertexIndex[g] = static_cast<int>(mesh.vertices.size());
                    camPoints[g] = Xc;
                    mesh.vertices.emplace_back(static_cast<float>(Xw[0]), static_cast<float>(Xw[1]), static_cast<float>(Xw[2]));
                    mesh.colors.push_back(color.at<cv::Vec3b>(v, u));
                }
            }

            auto addTriangle = [&](size_t a, size_t b, size_t c) {
                int ia = vertexIndex[a], ib = vertexIndex[b], ic = vertexIndex[c];
                if (ia < 0 || ib < 0 || ic < 0) return false;
                const cv::Vec3d &A = camPoints[a], &B = camPoints[b], &C = camPoints[c];
                double zMin = std::min({A[2], B[2], C[2]}), zMax = std::max({A[2], B[2], C[2]});
                if (zMax > zMin * (1.0 + m_maxDepthJump)) return false;    // розрив глибини

                cv::Vec3d n = (B - A).cross(C - A);
                double nn = cv::norm(n), na = cv::norm(A);
                if (nn < 1e-18) return false;
                double viewCos = n.dot(A) / (nn * na);
                if (std::abs(viewCos) < m_minViewCos) return false;         // трикутник "по дотичній"
                // Нормаль — у бік камери, щоб поверхня була коректно орієнтована
                if (viewCos > 0) mesh.faces.emplace_back(ia, ic, ib);
                else mesh.faces.emplace_back(ia, ib, ic);
                return true;
            };

            coverage[idx] = cv::Mat::zeros(D.size(), CV_32F);

            for (int r = 0; r + 1 < gridRows; ++r) {
                for (int c = 0; c + 1 < gridCols; ++c) {
                    size_t g00 = static_cast<size_t>(r) * gridCols + c, g01 = g00 + 1;
                    size_t g10 = g00 + gridCols, g11 = g10 + 1;
                    bool first = addTriangle(g00, g10, g01);
                    bool second = addTriangle(g01, g10, g11);
                    if (first || second) {
                        cv::Rect cell(c * m_step, r * m_step, m_step + 1, m_step + 1);
                        cell &= cv::Rect(0, 0, D.cols, D.rows);
                        D(cell).copyTo(coverage[idx](cell));
                    }
                }
            }

            std::cout << "[Mesh] Карта #" << maps[idx].refId << " (пара з #" << maps[idx].srcId << "): "
                      << mesh.vertices.size() - firstVertex << " вершин, "
                      << mesh.faces.size() - firstFace << " трикутників" << std::endl;
            processed.push_back(idx);
        }

        // Дрібні відокремлені шматки (шум, уривки фону) прибираємо: зв'язні компоненти за спільними вершинами
        std::vector<int> parent(mesh.vertices.size());
        std::iota(parent.begin(), parent.end(), 0);
        auto find = [&](int x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
        for (const auto& f : mesh.faces) {
            int a = find(f[0]), b = find(f[1]), c = find(f[2]);
            parent[a] = b; parent[find(c)] = find(b);
        }
        std::unordered_map<int, int> componentFaces;
        int largestComponent = 0;
        for (const auto& f : mesh.faces) largestComponent = std::max(largestComponent, ++componentFaces[find(f[0])]);
        const int minComponentFaces = std::max(m_minComponentFaces, largestComponent / 200);
        std::vector<cv::Vec3i> keptFaces;
        keptFaces.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces) {
            if (componentFaces[find(f[0])] >= minComponentFaces) keptFaces.push_back(f);
        }
        std::cout << "[Mesh] Видалено дрібних фрагментів: " << mesh.faces.size() - keptFaces.size() << " трикутників" << std::endl;
        mesh.faces.swap(keptFaces);

        // Вершини без жодного трикутника (поодинокі точки) не потрібні
        std::vector<int> remap(mesh.vertices.size(), -1);
        for (const auto& f : mesh.faces) for (int k = 0; k < 3; ++k) remap[f[k]] = 0;
        ColoredMesh compact;
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            if (remap[i] < 0) continue;
            remap[i] = static_cast<int>(compact.vertices.size());
            compact.vertices.push_back(mesh.vertices[i]);
            compact.colors.push_back(mesh.colors[i]);
        }
        compact.faces.reserve(mesh.faces.size());
        for (const auto& f : mesh.faces) compact.faces.emplace_back(remap[f[0]], remap[f[1]], remap[f[2]]);

        std::cout << "[Mesh] Поверхня: " << compact.vertices.size() << " вершин, "
                  << compact.faces.size() << " трикутників" << std::endl;
        return compact;
    }

    /**
     * Збереження сітки з кольорами вершин у бінарний .ply (відкривається в MeshLab, Blender, CloudCompare)
     */
    static bool savePlyMesh(const std::string& filename, const ColoredMesh& mesh) {
        std::ofstream ofs(filename, std::ios::binary);
        if (!ofs.is_open()) {
            std::cerr << "[Mesh] Не вдалося відкрити файл для запису: " << filename << std::endl;
            return false;
        }

        ofs << "ply\n";
        ofs << "format binary_little_endian 1.0\n";
        ofs << "element vertex " << mesh.vertices.size() << "\n";
        ofs << "property float x\n";
        ofs << "property float y\n";
        ofs << "property float z\n";
        ofs << "property uchar red\n";
        ofs << "property uchar green\n";
        ofs << "property uchar blue\n";
        ofs << "element face " << mesh.faces.size() << "\n";
        ofs << "property list uchar int vertex_indices\n";
        ofs << "end_header\n";

        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            ofs.write(reinterpret_cast<const char*>(mesh.vertices[i].val), 3 * sizeof(float));
            unsigned char rgb[3] = { mesh.colors[i][2], mesh.colors[i][1], mesh.colors[i][0] };
            ofs.write(reinterpret_cast<const char*>(rgb), sizeof(rgb));
        }
        for (const auto& f : mesh.faces) {
            unsigned char count = 3;
            ofs.write(reinterpret_cast<const char*>(&count), 1);
            ofs.write(reinterpret_cast<const char*>(f.val), 3 * sizeof(int));
        }

        std::cout << "[Mesh] Збережено: " << filename << std::endl;
        return true;
    }
};
