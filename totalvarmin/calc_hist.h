#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <random>
#include <tuple>
#include <vector>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/view/transform.hpp>

#include <sqr.h>

template <std::floating_point ElementType>
auto calc_hist(
    const Field<ElementType> &a
  , double 𝛿
  , uint16_t nbins)
{
    namespace rs = ranges;
    namespace rv = rs::views;

    using field      = Field<ElementType>;
    using value_type = field::value_type;

    auto rd = std::random_device{};

    auto hist = std::vector<uint64_t>(nbins, 0);
    auto hist_p = &hist[0];
    #pragma omp parallel
    {
        auto gen  = std::mt19937(rd());
        auto dist = std::uniform_real_distribution{-𝛿 / 2, 𝛿 / 2};
        #pragma omp for reduction(+:hist_p[:nbins])
        for (auto col = a.col_begin(); a.col_end() > col; ++col) {
            for (auto row = a.row_begin(col); a.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; a.n_images() > n; ++n) {
                    auto v = a[n, row, col];
                    v += dist(gen);
                    auto bin = int32_t(std::round(nbins * v - value_type(0.5)));
                    if (0 > bin) bin = 0;
                    if (nbins <= bin) bin = nbins - 1;
                    ++hist_p[bin];
                }
            }
        }
    }

    auto total = rs::accumulate(hist, uint64_t{}) / double(nbins);
    auto fp_hist = std::vector<double>(nbins, 0);
    rs::copy(rv::transform(hist, [&](auto v) { return v / total; }), fp_hist.begin());

    return fp_hist;
}
