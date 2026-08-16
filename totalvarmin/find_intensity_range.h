#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif

#include <range/v3/numeric/accumulate.hpp>

#include <layout_cylinder.h>

#include <sqr.h>

template <std::integral ElementType, typename Extents>
std::tuple<
    typename std::mdspan<ElementType, Extents, layout_cylinder>::value_type
  , typename std::mdspan<ElementType, Extents, layout_cylinder>::value_type
  >
find_intensity_range(
    std::mdspan<ElementType, Extents, layout_cylinder> a
  , double tail_thr)
{
    namespace rs = ranges;

    using mdspan     = std::mdspan<ElementType, Extents, layout_cylinder>;
    using value_type = mdspan::value_type;

    auto constexpr minv      = std::numeric_limits<value_type>::min();
    auto constexpr maxv      = std::numeric_limits<value_type>::max();
    auto constexpr hist_size = maxv - minv + 1;

    auto m = a.mapping();
    auto hist = std::vector<uint64_t>(hist_size, 0);
    auto hist_p = &hist[0];
    #pragma omp parallel for reduction(+:hist_p[:hist_size]) schedule(static, 1)
    for (auto col = m.col_begin(); m.col_end() > col; ++col) {
        for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
            for (auto n = uint32_t{}; m.n_images() > n; ++n) {
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
