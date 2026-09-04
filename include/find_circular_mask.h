#pragma once

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/geometry.hpp>

#include <range/v3/algorithm/max_element.hpp>

auto find_circular_mask(const cv::Mat &g) {
    namespace rs = ranges;

    auto img8 = cv::Mat{};
    cv::normalize(g, img8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    auto bin = cv::Mat{};
    cv::threshold(img8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    const auto white_border =
        cv::countNonZero(bin.row(0)) +
        cv::countNonZero(bin.row(bin.rows - 1)) +
        cv::countNonZero(bin.col(0)) +
        cv::countNonZero(bin.col(bin.cols - 1));

    if (white_border > bin.rows + bin.cols)
        cv::bitwise_not(bin, bin);

    auto contours = std::vector<std::vector<cv::Point>>{};
    cv::findContours(bin, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    auto largest = rs::max_element(
        contours
      , [](auto &&a, auto &&b) {
            return cv::contourArea(a) < cv::contourArea(b);
      });

    auto center = cv::Point2f{};
    auto radius = float{};
    cv::minEnclosingCircle(*largest, center, radius);

#if 0
    auto marked = cv::Mat{};
    cv::cvtColor(img8, marked, cv::COLOR_GRAY2BGR);
    cv::circle(marked, center, cvRound(radius),
               cv::Scalar{0, 0, 255}, 1, cv::LINE_AA);
    cv::drawMarker(marked, center, cv::Scalar{0, 255, 0},
                   cv::MARKER_CROSS, 15, 2, cv::LINE_AA);
    cv::imwrite("circular-mask.png", marked);
#endif

    return std::tuple{cv::Point2d{center.x, center.y}, double(radius)};
}
