#include <array>
#include <cstdint>
#include <cstdio>
#include <memory>

#include <boost/hana.hpp>
#include <boost/hana/ext/std/array.hpp>

#include <cub/block/block_reduce.cuh>

#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>

#include <cuda/std/array>
#include <cuda/std/cmath>

#include <fmt/printf.h>

#include <sqr.h>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include <chambolle.h>

namespace hana = boost::hana;
using namespace hana::literals;

static constexpr auto threads_per_block = 256;

struct div_kernel {
    template <typename Config, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , cuda::std::array<float *, 3> xi
      , float *v
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto offset = m(n, row, col);
                auto prev_img_ofs = index_type{};
                auto prev_row_ofs = index_type{};
                auto prev_col_ofs = index_type{};

                if (0 < n) {
                    prev_img_ofs = m(n - 1, row, col);
                }
                if (m.row_begin(col) < row) {
                    prev_row_ofs = m(n, row - 1, col) ;
                }
                if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1)) {
                    prev_col_ofs = m(n, row, col - 1);
                }

                auto d = 0.0f;

                if (0 < n)                d -= xi[0_c][prev_img_ofs];
                if (m.n_images() - 1 > n) d += xi[0_c][offset];

                if (m.row_begin(col) < row)   d -= xi[1_c][prev_row_ofs];
                if (m.row_end(col) - 1 > row) d += xi[1_c][offset];

                if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                    d -= xi[2_c][prev_col_ofs];
                if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                    d += xi[2_c][offset];

                v[offset] = d;
            }
        }
    }
};

struct grad_kernel {
    template <typename Config, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , const float *g
      , cuda::std::array<float *, 3> xi
      , float *v
      , float lambda
      , float tau
      , cuda::std::span<float> g_maxDq
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;

        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        __shared__ typename BlockReduce::TempStorage temp;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto Δq  = 0.0f;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto offset = m(n, row, col);
                auto next_img_ofs = index_type{};
                auto next_row_ofs = index_type{};
                auto next_col_ofs = index_type{};

                if (m.n_images() - 1 > n) {
                    next_img_ofs = m(n + 1, row, col);
                }
                if (m.row_end(col) - 1 > row) {
                    next_row_ofs = m(n, row + 1, col);
                }
                if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                    next_col_ofs = m(n, row, col + 1);
                }

                auto g1 = 0.0f;
                if (m.n_images() - 1 > n) {
                    g1 = (v[next_img_ofs] - v[offset]) -
                         (g[next_img_ofs] - g[offset]) / lambda;
                }

                auto g2 = 0.0f;
                if (m.row_end(col) - 1 > row) {
                    g2 = (v[next_row_ofs] - v[offset]) -
                         (g[next_row_ofs] - g[offset]) / lambda;
                }

                auto g3 = 0.0f;
                if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                    g3 = (v[next_col_ofs] - v[offset]) -
                         (g[next_col_ofs] - g[offset]) / lambda;
                }

                auto denom = 1.0f + tau * cuda::std::sqrt(sqr(g1) + sqr(g2) + sqr(g3));
                auto &q1 = xi[0_c][offset];
                auto &q2 = xi[1_c][offset];
                auto &q3 = xi[2_c][offset];

                auto next_q1 = (q1 + tau * g1) / denom;
                auto next_q2 = (q2 + tau * g2) / denom;
                auto next_q3 = (q3 + tau * g3) / denom;

                Δq = sqr(q1 - next_q1) + sqr(q2 - next_q2) + sqr(q3 - next_q3);

                q1 = next_q1;
                q2 = next_q2;
                q3 = next_q3;
            }
        }
        auto block_max = BlockReduce(temp).Reduce(Δq, cuda::maximum<>{});
        if (0 == cuda::gpu_thread.rank(cuda::block, config)) {
            cuda::atomic_ref<float, cuda::thread_scope_device>{g_maxDq[0]}
                .fetch_max(block_max, cuda::std::memory_order_relaxed);
        }
    }
};

struct update_kernel {
    template <typename Config, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , float *g
      , cuda::std::array<float *, 3> xi
      , float lambda
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto offset = m(n, row, col);
                auto prev_img_ofs = index_type{};
                auto prev_row_ofs = index_type{};
                auto prev_col_ofs = index_type{};

                if (0 < n) {
                    prev_img_ofs = m(n - 1, row, col);
                }
                if (m.row_begin(col) < row) {
                    prev_row_ofs = m(n, row - 1, col) ;
                }
                if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1)) {
                    prev_col_ofs = m(n, row, col - 1);
                }

                auto d = 0.0f;

                if (0 < n)                d -= xi[0_c][prev_img_ofs];
                if (m.n_images() - 1 > n) d += xi[0_c][offset];

                if (m.row_begin(col) < row)   d -= xi[1_c][prev_row_ofs];
                if (m.row_end(col) - 1 > row) d += xi[1_c][offset];

                if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                    d -= xi[2_c][prev_col_ofs];
                if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                    d += xi[2_c][offset];

                g[offset] = g[offset] - lambda * d;
            }
        }
    }
};

void chambolle(CudaContext &ctx, DeviceField<float> &g, float 𝜆, float 𝜏, int iters) {
    auto mapping = g.mapping();

    auto Ξ = hana::unpack(hana::make_range(0_c, hana::llong_c<3>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, 3>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), mapping.required_span_size(), cuda::no_init})...
        }};
    });
    auto v = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), mapping.required_span_size(), cuda::no_init};

    for (auto &&𝜉 : Ξ) {
        cuda::fill_bytes(ctx.stream(), 𝜉, 0.0f);
    }
    cuda::fill_bytes(ctx.stream(), v, 0.0f);

    auto Ξ_ptrs = hana::unpack(Ξ, [](auto&... buffer) {
        return cuda::std::array<float*, 3>{{buffer.data()...}};
    });
    auto v_ptr = v.data();

    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            (mapping.nstride() + (threads_per_block - 1)) / threads_per_block
          , mapping.extents().extent(1)
          , mapping.col_end() - mapping.col_begin()
          })
      , cuda::block_dims<threads_per_block>());

    auto d_maxΔq = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), 1, cuda::no_init};

    // auto f = std::unique_ptr<FILE, decltype(&fclose)>{
    //     fopen("maxΔq.txt", "wt")
    //   , &fclose
    //   };
    // fmt::print(f.get(), "iter dqmax\n");

    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        cuda::launch(ctx.stream(), config, div_kernel{}, mapping, Ξ_ptrs, v_ptr);
        cuda::fill_bytes(ctx.stream(), d_maxΔq, 0);
        cuda::launch(ctx.stream(), config, grad_kernel{}, mapping, g.data(), Ξ_ptrs, v_ptr, 𝜆, 𝜏, d_maxΔq);

        auto maxΔq = 0.0f;
        cuda::copy_bytes(ctx.stream(), d_maxΔq, cuda::std::span<float>{&maxΔq, 1});
        ctx.stream().sync();

        fmt::print("{:4} {:10.8f}\r", n_iter, std::sqrt(maxΔq));
        fflush(stdout);
        // fmt::print(f.get(), "{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
    }
    cuda::launch(ctx.stream(), config, update_kernel{}, mapping, g.data(), Ξ_ptrs, 𝜆);
    fmt::print("\n");
}
