#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <tuple>
#include <vector>

#include <fmt/printf.h>

#include <sqr.h>

template <std::floating_point ElementType>
void chambolle(
    Field<ElementType> &g
  , typename Field<ElementType>::value_type 𝜆
  , typename Field<ElementType>::value_type 𝜏
  , int iters)
{
    using field      = Field<ElementType>;
    using value_type = field::value_type;

    auto p1    = Field<float>(g.mapping());
    auto p2    = Field<float>(g.mapping());
    auto p3    = Field<float>(g.mapping());
    auto div_p = Field<float>(g.mapping());

    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        #pragma omp parallel for schedule(static, 1)
        for (auto col = g.col_begin(); g.col_end() > col; ++col) {
            for (auto row = g.row_begin(col); g.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; g.n_images() > n; ++n) {
                    auto d = value_type{};
                    if (0 < n)                d -= p1[n - 1, row, col];
                    if (g.n_images() - 1 > n) d += p1[n, row, col];

                    if (g.row_begin(col) < row)   d -= p2[n, row - 1, col];
                    if (g.row_end(col) - 1 > row) d += p2[n, row, col];

                    if (g.col_begin() < col && g.row_begin(col - 1) <= row && row < g.row_end(col - 1))
                        d -= p3[n, row, col - 1];
                    if (g.col_end() - 1 > col && g.row_begin(col + 1) <= row && row < g.row_end(col + 1))
                        d += p3[n, row, col];

                    div_p[n, row, col] = d;
                }
            }
        }

        auto maxΔq = 0.0;
        #pragma omp parallel for reduction(max:maxΔq) schedule(static, 1)
        for (auto col = g.col_begin(); g.col_end() > col; ++col) {
            for (auto row = g.row_begin(col); g.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; g.n_images() > n; ++n) {
                    auto g1 = value_type{};
                    if (g.n_images() - 1 > n) {
                        g1 = (div_p[n + 1, row, col] - div_p[n, row, col]) -
                             (g[n + 1, row, col] - g[n, row, col]) / 𝜆;
                    }

                    auto g2 = value_type{};
                    if (g.row_end(col) - 1 > row) {
                        g2 = (div_p[n, row + 1, col] - div_p[n, row, col]) -
                             (g[n, row + 1, col] - g[n, row, col]) / 𝜆;
                    }

                    auto g3 = value_type{};
                    if (g.col_end() - 1 > col && g.row_begin(col + 1) <= row && row < g.row_end(col + 1)) {
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
    for (auto col = g.col_begin(); g.col_end() > col; ++col) {
        for (auto row = g.row_begin(col); g.row_end(col) > row; ++row) {
            for (auto n = uint32_t{}; g.n_images() > n; ++n) {
                auto d = value_type{};
                if (0 < n)                d -= p1[n - 1, row, col];
                if (g.n_images() - 1 > n) d += p1[n, row, col];

                if (g.row_begin(col) < row)   d -= p2[n, row - 1, col];
                if (g.row_end(col) - 1 > row) d += p2[n, row, col];

                if (g.col_begin() < col && g.row_begin(col - 1) <= row && row < g.row_end(col - 1))
                    d -= p3[n, row, col - 1];
                if (g.col_end() - 1 > col && g.row_begin(col + 1) <= row && row < g.row_end(col + 1))
                    d += p3[n, row, col];

                g[n, row, col] = g[n, row, col] - 𝜆 * d;
            }
        }
    }
}
