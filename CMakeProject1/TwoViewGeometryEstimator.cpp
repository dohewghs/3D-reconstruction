#include "TwoViewGeometryEstimator.hpp"
#include <iostream>

TwoViewReconstructionResult TwoViewGeometryEstimator::reconstruct(
    const Frame& frame1, 
    const Frame& frame2, 
    const std::vector<cv::DMatch>& matches) 
{
    TwoViewReconstructionResult result;

    if (matches.size() < 8) {
        std::cerr << "Недостатньо відповідностей для оцінки геометрії двох видів." << std::endl;
        return result;
    }

    // 1. Витягування ключових точок
    std::vector<cv::Point2f> pointsPrev, pointsCurr;
    pointsPrev.reserve(matches.size());
    pointsCurr.reserve(matches.size());

    for (const auto& m : matches) {
        pointsPrev.push_back(frame1.keypoints[m.queryIdx].pt);
        pointsCurr.push_back(frame2.keypoints[m.trainIdx].pt);
    }

    // 2. Оцінка есенціальної матриці
    cv::Mat inlierMask;
    cv::Mat K = this->m_cameraMatrix;
    K.convertTo(K, CV_64F);

    cv::Mat E = cv::findEssentialMat(
        pointsPrev, 
        pointsCurr, 
        K, 
        cv::RANSAC, 
        this->m_ransacProb, 
        this->m_ransacThreshold, 
        inlierMask
    );

    if (E.empty()) {
        std::cerr << "Не вдалося оцінити есенціальну матрицю." << std::endl;
        return result;
    }

    // 3. Відновлення пози (R, t)
    cv::Mat R1, t;
    cv::recoverPose(E, pointsPrev, pointsCurr, K, R1, t, inlierMask);
    R1.convertTo(R1, CV_64F);
    t.convertTo(t, CV_64F);

    // 4. Створення проєкційних матриць 3x4
    cv::Mat P0 = cv::Mat::zeros(3, 4, CV_64F);
    K.copyTo(P0(cv::Rect(0, 0, 3, 3)));

    cv::Mat Rt = cv::Mat::zeros(3, 4, CV_64F);
    R1.copyTo(Rt(cv::Rect(0, 0, 3, 3)));
    t.copyTo(Rt(cv::Rect(3, 0, 1, 3))); // безпечне копіювання вектора t
    cv::Mat P1 = K * Rt;

    // 5. Витягування інлаєрів через прямий показчик на пам'ять маски
    std::vector<cv::Point2f> inlierPointsPrev, inlierPointsCurr;
    std::vector<int> candidateMatchIndices;

    if (inlierMask.type() != CV_8U) {
        inlierMask.convertTo(inlierMask, CV_8U);
    }
    
    // Гарантуємо безперервність пам'яті маски
    cv::Mat contMask = inlierMask.isContinuous() ? inlierMask : inlierMask.clone();
    const uchar* maskPtr = contMask.ptr<uchar>();

    for (size_t i = 0; i < matches.size(); ++i) {
        if (maskPtr[i]) {
            inlierPointsPrev.push_back(pointsPrev[i]);
            inlierPointsCurr.push_back(pointsCurr[i]);
            candidateMatchIndices.push_back(static_cast<int>(i));
        }
    }

    if (inlierPointsPrev.empty()) {
        return result;
    }

    // 6. Тріангуляція точок
    cv::Mat points4D;
    cv::triangulatePoints(P0, P1, inlierPointsPrev, inlierPointsCurr, points4D);

    // 7. Перетворення у 3D з адаптивним зчитуванням типу (float або double)
    result.points3D.reserve(points4D.cols);
    result.inlierMatchIndices.reserve(points4D.cols);

    const bool isDouble = (points4D.type() == CV_64F || points4D.depth() == CV_64F);

    for (int i = 0; i < points4D.cols; ++i) {
        double w = isDouble ? points4D.at<double>(3, i) : points4D.at<float>(3, i);

        if (std::abs(w) <= 1e-6) {
            continue;
        }

        double x = (isDouble ? points4D.at<double>(0, i) : points4D.at<float>(0, i)) / w;
        double y = (isDouble ? points4D.at<double>(1, i) : points4D.at<float>(1, i)) / w;
        double z = (isDouble ? points4D.at<double>(2, i) : points4D.at<float>(2, i)) / w;

        // Перевірка глибини спереду камери
        if (z > 0.0 && z < 100.0) {
            result.points3D.emplace_back(x, y, z);
            result.inlierMatchIndices.push_back(candidateMatchIndices[i]);
        }
    }

    // 8. Заповнення фінального результату
    result.success = !result.points3D.empty();
    result.R = R1;
    result.t = t;
    result.inliersPrev = inlierPointsPrev;
    result.inliersCurr = inlierPointsCurr;
    result.inlierMask = contMask;

    return result;
}