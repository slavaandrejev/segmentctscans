#pragma once

#include <algorithm>
#include <span>
#include <vector>

#include <range/v3/algorithm/stable_sort.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/iota.hpp>

// Project `a` on a set where all pair-wise distances are above or equal 𝜀
auto gap_set_project(std::span<const double> a, double 𝜀) {
    namespace rs = ranges;
    namespace rv = rs::views;

    const auto n = a.size();
    auto labels = std::vector<double>(n);
    if (n == 0) return labels;

    const auto upper = 1.0 - 𝜀 * double(n - 1);

    auto order = rv::iota(size_t{}, n) | rs::to_vector;
    rs::stable_sort(order, [&](auto i, auto j) { return a[i] < a[j]; });

    struct Block {
        double sum;
        std::size_t count;
    };

    auto blocks = std::vector<Block>{};
    blocks.reserve(n);

    for (auto i = size_t{}; n > i; ++i) {
        blocks.push_back({a[order[i]] - 𝜀 * double(i), 1});

        while (blocks.size() > 1) {
            auto &left = blocks[blocks.size() - 2];
            const auto &right = blocks.back();

            if (left.sum / double(left.count) <= right.sum / double(right.count)) {
                break;
            }

            left.sum += right.sum;
            left.count += right.count;
            blocks.pop_back();
        }
    }

    auto i = std::size_t{};
    for (const auto &block : blocks) {
        const auto value = std::clamp(
            block.sum / double(block.count), 0.0, upper);

        for (std::size_t j = 0; j < block.count; ++j, ++i) {
            labels[order[i]] = value + 𝜀 * double(i);
        }
    }

    return labels;
}
