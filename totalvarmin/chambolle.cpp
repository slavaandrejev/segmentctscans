#include <cstdint>
#include <cstdio>
#include <memory>

#include <cub/block/block_reduce.cuh>

#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/std/cmath>

#include <fmt/printf.h>

#include <sqr.h>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include <chambolle.h>

static constexpr auto threads_per_block = 256;

struct div_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , const mf32_span p1
      , const mf32_span p2
      , const mf32_span p3
      ,       mf32_span div_p
      )
    {
        auto m = p1.mapping();

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto d = 0.0f;

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
};

struct grad_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , const mf32_span g
      , const mf32_span p1
      , const mf32_span p2
      , const mf32_span p3
      ,       mf32_span div_p
      , float 𝜆
      , float 𝜏
      , float *g_maxΔq
      )
    {
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        __shared__ typename BlockReduce::TempStorage temp;

        auto m = p1.mapping();

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto Δq  = 0.0f;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto g1 = 0.0f;
                if (m.n_images() - 1 > n) {
                    g1 = (div_p[n + 1, row, col] - div_p[n, row, col]) -
                         (g[n + 1, row, col] - g[n, row, col]) / 𝜆;
                }

                auto g2 = 0.0f;
                if (m.row_end(col) - 1 > row) {
                    g2 = (div_p[n, row + 1, col] - div_p[n, row, col]) -
                         (g[n, row + 1, col] - g[n, row, col]) / 𝜆;
                }

                auto g3 = 0.0f;
                if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                    g3 = (div_p[n, row, col + 1] - div_p[n, row, col]) -
                         (g[n, row, col + 1] - g[n, row, col]) / 𝜆;
                }

                auto denom = 1.0f + 𝜏 * cuda::std::sqrt(sqr(g1) + sqr(g2) + sqr(g3));
                auto &q1 = p1[n, row, col];
                auto &q2 = p2[n, row, col];
                auto &q3 = p3[n, row, col];

                auto next_q1 = (q1 + 𝜏 * g1) / denom;
                auto next_q2 = (q2 + 𝜏 * g2) / denom;
                auto next_q3 = (q3 + 𝜏 * g3) / denom;

                Δq = sqr(q1 - next_q1) + sqr(q2 - next_q2) + sqr(q3 - next_q3);

                q1 = next_q1;
                q2 = next_q2;
                q3 = next_q3;
            }
        }
        auto block_max = BlockReduce(temp).Reduce(Δq, cuda::maximum<>{});
        if (0 == cuda::gpu_thread.rank(cuda::block, config)) {
            cuda::atomic_ref<float, cuda::thread_scope_device>{*g_maxΔq}
                .fetch_max(block_max, cuda::std::memory_order_relaxed);
        }
    }
};

struct update_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      ,       mf32_span g
      , const mf32_span p1
      , const mf32_span p2
      , const mf32_span p3
      , float 𝜆
      )
    {
        auto m = p1.mapping();

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto d = 0.0f;

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
};

void chambolle(CudaContext &ctx, DeviceField<float> &g, float 𝜆, float 𝜏, int iters) {
    auto mapping = g.mapping();

    auto p1    = DeviceField<float>{mapping, ctx.stream(), ctx.mr()};
    auto p2    = DeviceField<float>{mapping, ctx.stream(), ctx.mr()};
    auto p3    = DeviceField<float>{mapping, ctx.stream(), ctx.mr()};
    auto div_p = DeviceField<float>{mapping, ctx.stream(), ctx.mr()};

    auto p1_view    = p1.view();
    auto p2_view    = p2.view();
    auto p3_view    = p3.view();
    auto div_p_view = div_p.view();

    cuda::fill_bytes(
        ctx.stream()
      , cuda::std::span{
            p1_view.data_handle()
          , p1_view.data_handle()  + mapping.required_span_size()
          }
      , 0);
    cuda::fill_bytes(
        ctx.stream()
      , cuda::std::span{
            p2_view.data_handle()
          , p2_view.data_handle()  + mapping.required_span_size()
          }
      , 0);
    cuda::fill_bytes(
        ctx.stream()
      , cuda::std::span{
            p3_view.data_handle()
          , p3_view.data_handle()  + mapping.required_span_size()
          }
      , 0);

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
        cuda::launch(ctx.stream(), config, div_kernel{}, p1_view, p2_view, p3_view, div_p_view);
        cuda::fill_bytes(ctx.stream(), d_maxΔq, 0);
        cuda::launch(ctx.stream(), config, grad_kernel{}, g.view(), p1_view, p2_view, p3_view, div_p_view, 𝜆, 𝜏, d_maxΔq.data());

        auto maxΔq = 0.0f;
        cuda::copy_bytes(ctx.stream(), d_maxΔq, cuda::std::span<float>{&maxΔq, 1});
        ctx.stream().sync();

        fmt::print("{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
        // fmt::print(f.get(), "{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
    }
    cuda::launch(ctx.stream(), config, update_kernel{}, g.view(), p1_view, p2_view, p3_view, 𝜆);
}
