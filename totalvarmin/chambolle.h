#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <tuple>
#include <vector>

#include <fmt/printf.h>

#include <sqr.h>

template <std::floating_point ElementType, typename Extents>
void chambolle(
    std::mdspan<ElementType, Extents, layout_cylinder> g
  , typename Field<ElementType>::value_type 𝜆
  , typename Field<ElementType>::value_type 𝜏
  , int iters)
{
    using field      = Field<ElementType>;
    using value_type = field::value_type;

    auto m = g.mapping();

    auto p1_field    = Field<float>(m);
    auto p2_field    = Field<float>(m);
    auto p3_field    = Field<float>(m);
    auto div_p_field = Field<float>(m);

    auto p1    = p1_field.view();
    auto p2    = p2_field.view();
    auto p3    = p3_field.view();
    auto div_p = div_p_field.view();

    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        #pragma omp parallel for schedule(static, 1)
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                    auto d = value_type{};
                    if (0 < n)                d -= p1[n - 1, row, col];
                    if (m.n_images() - 1 > n) d += p1[n, row, col];

                    if (m.row_begin(col) < row)   d -= p2[n, row - 1, col];
                    if (m.row_end(col) - 1 > row) d += p2[n, row, col];

                    if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                        d -= p3[n, row, col - 1];
                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                        d += p3[n, row, col];

                    div_p[n, row, col] = d;
                }
            }
        }

        auto maxΔq = 0.0;
        #pragma omp parallel for reduction(max:maxΔq) schedule(static, 1)
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                    auto g1 = value_type{};
                    if (m.n_images() - 1 > n) {
                        g1 = (div_p[n + 1, row, col] - div_p[n, row, col]) -
                             (g[n + 1, row, col] - g[n, row, col]) / 𝜆;
                    }

                    auto g2 = value_type{};
                    if (m.row_end(col) - 1 > row) {
                        g2 = (div_p[n, row + 1, col] - div_p[n, row, col]) -
                             (g[n, row + 1, col] - g[n, row, col]) / 𝜆;
                    }

                    auto g3 = value_type{};
                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                        g3 = (div_p[n, row, col + 1] - div_p[n, row, col]) -
                             (g[n, row, col + 1] - g[n, row, col]) / 𝜆;
                    }

                    auto denom = 1.0 + 𝜏 * std::sqrt(sqr(g1) + sqr(g2) + sqr(g3));
                    auto &q1 = p1[n, row, col];
                    auto &q2 = p2[n, row, col];
                    auto &q3 = p3[n, row, col];

                    auto next_q1 = (q1 + 𝜏 * g1) / denom;
                    auto next_q2 = (q2 + 𝜏 * g2) / denom;
                    auto next_q3 = (q3 + 𝜏 * g3) / denom;

                    auto Δq = sqr(q1 - next_q1) + sqr(q2 - next_q2) + sqr(q3 - next_q3);
                    maxΔq = std::max(maxΔq, Δq);

                    q1 = next_q1;
                    q2 = next_q2;
                    q3 = next_q3;
                }
            }
        }
        fmt::print("{} {}\n", n_iter, std::sqrt(maxΔq));
    }

    #pragma omp parallel for schedule(static, 1)
    for (auto col = m.col_begin(); m.col_end() > col; ++col) {
        for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
            for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                auto d = value_type{};
                if (0 < n)                d -= p1[n - 1, row, col];
                if (m.n_images() - 1 > n) d += p1[n, row, col];

                if (m.row_begin(col) < row)   d -= p2[n, row - 1, col];
                if (m.row_end(col) - 1 > row) d += p2[n, row, col];

                if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                    d -= p3[n, row, col - 1];
                if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                    d += p3[n, row, col];

                g[n, row, col] = g[n, row, col] - 𝜆 * d;
            }
        }
    }
}
