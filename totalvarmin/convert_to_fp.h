#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif

#include <boost/align/aligned_allocator.hpp>

#include <field.h>

template <std::integral ElementType, typename Extents>
auto convert_to_fp(
    std::mdspan<ElementType, Extents, layout_cylinder> a
  , std::remove_cv_t<ElementType> lo
  , std::remove_cv_t<ElementType> hi)
{
    auto m = a.mapping();
    auto img = Field<float>(m);
    auto img_span = img.view();

    #pragma omp parallel for schedule(static, 1)
    for (auto col = m.col_begin(); m.col_end() > col; ++col) {
        for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
            for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                auto v = a[n, row, col];
                if (v <= lo) {
                    img_span[n, row, col] = 0.0f;
                    continue;
                }
                if (v >= hi) {
                    img_span[n, row, col] = 1.0f;
                    continue;
                }
                img_span[n, row, col] = float(v - lo) / (hi - lo);
            }
        }
    }

    return img;
}
