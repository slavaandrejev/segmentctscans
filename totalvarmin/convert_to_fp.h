#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

template <std::integral ElementType>
auto convert_to_fp(
    const Field<ElementType> &a
  , std::remove_cv_t<ElementType> lo
  , std::remove_cv_t<ElementType> hi)
{
    auto img = Field<float>(a.mapping());

    #pragma omp parallel for schedule(static, 1)
    for (auto col = a.col_begin(); a.col_end() > col; ++col) {
        for (auto row = a.row_begin(col); a.row_end(col) > row; ++row) {
            for (auto n = uint32_t{}; a.n_images() > n; ++n) {
                auto v = a[n, row, col];
                if (v <= lo) {
                    img[n, row, col] = 0.0f;
                    continue;
                }
                if (v >= hi) {
                    img[n, row, col] = 1.0f;
                    continue;
                }
                img[n, row, col] = float(v - lo) / (hi - lo);
            }
        }
    }

    return img;
}
