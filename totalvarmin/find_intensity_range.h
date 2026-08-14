#pragma once

#include <cmath>
#include <concepts>
#include <limits>
#include <tuple>
#include <vector>

#include <range/v3/numeric/accumulate.hpp>

#include <field.h>

#include <sqr.h>

template <std::integral ElementType>
auto find_intensity_range(
    const Field<ElementType> &a
  , double tail_thr)
{
    namespace rs = ranges;

    using field      = Field<ElementType>;
    using value_type = field::value_type;

    auto constexpr minv      = std::numeric_limits<value_type>::min();
    auto constexpr maxv      = std::numeric_limits<value_type>::max();
    auto constexpr hist_size = maxv - minv + 1;

    auto hist = std::vector<uint64_t>(hist_size, 0);
    auto hist_p = &hist[0];
    #pragma omp parallel for reduction(+:hist_p[:hist_size]) schedule(static, 1)
    for (auto col = a.col_begin(); a.col_end() > col; ++col) {
        for (auto row = a.row_begin(col); a.row_end(col) > row; ++row) {
            for (auto n = uint32_t{}; a.n_images() > n; ++n) {
                auto v = a[n, row, col];
                ++hist_p[v - minv];
            }
        }
    }

    auto total = rs::accumulate(hist, uint64_t{});

    auto const pct_lo = tail_thr * total;
    auto const pct_hi = (1.0 - tail_thr) * total;
    auto lo = size_t{}, hi = hist.size();
    auto cum = uint64_t{};
    for (auto i = size_t{}; hist.size() > i; ++i) {
        cum += hist[i];
        if (pct_lo >= cum) { lo = i; }
        if (pct_hi <= cum) { hi = i; break; }
    }

    return std::tuple{value_type(lo + minv), value_type(hi + minv)};
}
