#include "TwoViewGeometryEstimator.hpp"

TwoViewReconstructionResult TwoViewGeometryEstimator::reconstruct(const Frame& frame1, 
    const Frame& frame2, 
    const std::vector<cv::DMatch>& matches) {

    TwoViewReconstructionResult result;

    // Перевірка наявності достатньої кількості відповідностей
    if (matches.size() < 8) {
        std::cerr << "Недостатньо відповідностей для оцінки геометрії двох видів." << std::endl;
        return result; // Повертаємо невдалий результат
    }

    // 1. Витягування ключових точок з кадрів
    std::vector<cv::Point2f> pointsPrev, pointsCurr;
    for (const auto& m : matches) {
        pointsPrev.push_back(frame1.keypoints[m.queryIdx].pt);
        pointsCurr.push_back(frame2.keypoints[m.trainIdx].pt);
    }

    // 2. Оцінка есенціальної матриці з використанням RANSAC
    cv::Mat inlierMask;
    cv::Mat E = cv::findEssentialMat(
        pointsPrev, 
        pointsCurr, 
        this->m_cameraMatrix, 
        cv::RANSAC, 
        this->m_ransacProb, 
        this->m_ransacThreshold, 
        inlierMask
    );

    if (E.empty()) {
        std::cerr << "Не вдалося оцінити есенціальну матрицю." << std::endl;
        return result; // Повертаємо невдалий результат
    }

    // 3. Розкладання Essential матриці для отримання можливих ротацій та трансляцій
    cv::Mat R1, t;
    cv::Mat K = this->m_cameraMatrix;

    cv::recoverPose(E, pointsPrev, pointsCurr, K, R1, t, inlierMask);

    // 4. Створення проєкційних матриць для двох кадрів
    cv::Mat P0 = K * cv::Mat::eye(3, 4, CV_64F); // Проєкційна матриця для першого кадру

    cv::Mat Rt = cv::Mat::zeros(3, 4, CV_64F);
    R1.copyTo(Rt(cv::Rect(0, 0, 3,3)));
    t.copyTo(Rt.col(3));
    cv::Mat P1 = K * Rt; // Проєкційна матриця для другого кадру

    // 5. Витягування індексів внутрішніх точок
    std::vector<cv::Point2f> inlierPointsPrev, inlierPointsCurr;
    for (size_t i = 0; i < matches.size(); ++i) {
        if (inlierMask.at<uchar>(i)) {
            inlierPointsPrev.push_back(pointsPrev[i]);
            inlierPointsCurr.push_back(pointsCurr[i]);
        }
    }

    // 6. Тріангуляція точок
    cv::Mat points4D;
    cv::triangulatePoints(P0, P1, inlierPointsPrev, inlierPointsCurr, points4D);

    // 7. Перетворення гомогенних координат у 3D
    std::vector<cv::Point3d> points3D;
    points3D.reserve(points4D.cols);
    for (int i = 0; i < points4D.cols; ++i) {
        double w = points4D.at<float>(3, i);

        if (std::abs(w) <= 1e-6) {
            continue; // Пропускаємо цю точку
        }

        double x = points4D.at<float>(0, i) / w;
        double y = points4D.at<float>(1, i) / w;
        double z = points4D.at<float>(2, i) / w;

        if (z < 0) {
            continue; // Пропускаємо цю точку
        }

        points3D.emplace_back(x, y, z);
    }

    // 8. Заповнення результату
    result.success = !result.points3D.empty();
    result.R = R1;
    result.t = t;
    result.points3D = points3D;
    result.inliersPrev = inlierPointsPrev;
    result.inliersCurr = inlierPointsCurr;
    result.inlierMask = inlierMask;

    return result;
}