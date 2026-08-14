#pragma once

#include <tuple>
#include <vector>

#include <opencv2/opencv.hpp>

#include <range/v3/algorithm/max_element.hpp>

auto find_circular_mask(const cv::Mat &g) {
    namespace rs = ranges;

    auto img8 = cv::Mat{};
    cv::normalize(g, img8, 0, 255, cv::NORM_MINMAX, CV_8UC1);

    auto bin = cv::Mat{};
    cv::threshold(img8, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

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

    return std::tuple{cv::Point2d{center.x, center.y}, double(radius)};
}
