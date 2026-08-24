#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

#include <cub/block/block_reduce.cuh>
#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>

#include <Eigen/Dense>

#include <fmt/printf.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <sqr.h>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include "potts-min-partition.h"

namespace rs = ranges;
namespace rv = rs::views;

using namespace Eigen;

static constexpr auto threads_per_block = 256;
static constexpr auto max_k = 6;

struct init_v_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , const mf32_span g
      , cuda::std::span<mf32_span> V
      , cuda::std::span<mf32_span> V_bar
      , const cuda::std::span<float> ci
      )
    {
        auto m   = V[0].mapping();
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto v = g[n, row, col];
                auto dens = (Map<ArrayXf>{&ci[0], Index(ci.size())} - ArrayXf::Constant(ci.size(), v)).abs();
                auto cluster = Index{};
                dens.minCoeff(&cluster);
                for (auto i = 0; int(V.size()) > i; ++i) {
                    V_bar[i][n, row, col] = V[i][n, row, col] = i == cluster ? 1.0f : 0.0f;
                }
            }
        }
    }
};

struct grad_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , cuda::std::span<mf32_span> xi
      , cuda::std::span<mf32_span> V_bar
      , float 𝜏2
      )
    {
        const auto k = int(V_bar.size()) + 1;

        auto m   = xi[0].mapping();
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto q  = Matrix<float, 3, Dynamic, ColMajor, 3, max_k>{3, k};
                auto vk = 1.0f, vk_n = 1.0f, vk_r = 1.0f, vk_c = 1.0f;
                auto vi = 0.0f, vi_n = 0.0f, vi_r = 0.0f, vi_c = 0.0f;
                for (auto i = 0; k > i; ++i) {
                    if (k - 1 > i) {
                        vi = V_bar[i][n, row, col];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    q(0, i) = 0;
                    if (m.n_images() - 1 > n) {
                        if (k - 1 > i) {
                            vi_n = V_bar[i][n + 1, row, col];
                            vk_n -= vi_n;
                        } else {
                            vi_n = vk_n;
                        }
                        q(0, i) = vi_n - vi;
                    }

                    q(1, i) = 0.0f;
                    if (m.row_end(col) - 1 > row) {
                        if (k - 1 > i) {
                            vi_r = V_bar[i][n, row + 1, col];
                            vk_r -= vi_r;
                        } else {
                            vi_r = vk_r;
                        }
                        q(1, i) = vi_r - vi;
                    }

                    q(2, i) = 0.0f;
                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                        if (k - 1 > i) {
                            vi_c = V_bar[i][n, row, col + 1];
                            vk_c -= vi_c;
                        } else {
                            vi_c = vk_c;
                        }
                        q(2, i) = vi_c - vi;
                    }

                    q(0, i) *= 𝜏2;
                    q(0, i) += xi[i * 3][n, row, col];

                    q(1, i) *= 𝜏2;
                    q(1, i) += xi[i * 3 + 1][n, row, col];

                    q(2, i) *= 𝜏2;
                    q(2, i) += xi[i * 3 + 2][n, row, col];
                }
                Vector3f qi1i2[max_k][max_k];
                for (auto i1 = 0; k - 1 > i1; ++i1) {
                    for (auto i2 = i1 + 1; k > i2; ++i2) {
                        qi1i2[i1][i2] = Vector3f::Zero();
                    }
                }
                auto prev_q = Matrix<float, 3, Dynamic, ColMajor, 3, max_k>{3, k};
                for (auto sweep = 0; 20 > sweep; ++sweep) {
                    prev_q = q;
                    for (auto i1 = 0; k - 1 > i1; ++i1) {
                        for (auto i2 = i1 + 1; k > i2; ++i2) {
                            auto qh = (q.col(i2) - q.col(i1) + qi1i2[i1][i2]).eval();
                            auto l = qh.norm();
                            auto qhh = l > 1 ? ((l - 1.0f) * qh / l).eval() : Vector3f::Zero();
                            auto 𝛿 = 0.5 * (qhh - qi1i2[i1][i2]);
                            q.col(i1) += 𝛿;
                            q.col(i2) -= 𝛿;
                            qi1i2[i1][i2] = qhh;
                        }
                    }
                    if ((q - prev_q).array().abs().maxCoeff() < 1e-6) {
                        break;
                    }
                }
                for (auto i = 0; k > i; ++i) {
                    xi[i * 3][n, row, col]     = q(0, i);
                    xi[i * 3 + 1][n, row, col] = q(1, i);
                    xi[i * 3 + 2][n, row, col] = q(2, i);
                }
            }
        }
    }
};

struct div_kernel {
    template <typename Derived>
    static
    __device__ void project_simplex(MatrixBase<Derived> &q)
    {
        static_assert(1 == MatrixBase<Derived>::ColsAtCompileTime);

        auto u = q.eval();

        // insertion sort
        for (auto i = 1; u.rows() > i; ++i) {
            auto key = u[i]; auto j = i - 1;
            for (; 0 <= j && u[j] < key; --j) u[j + 1] = u[j];
            u[j + 1] = key;
        }

        auto sum   = 0.0f;
        auto 𝜌_sum = 0.0f;
        auto 𝜌     = 0;

        for (auto i = 0; u.rows() > i; ++i) {
            sum += u[i];

            const int j = i + 1;
            const float theta = (sum - 1.0f) / j;

            if (u[i] > theta) {
                𝜌 = j;
                𝜌_sum = sum;
            }
        }

        const float theta = (𝜌_sum - 1.0f) / 𝜌;

        for (int i = 0; i < q.rows(); ++i)
            q[i] = cuda::std::max(q[i] - theta, 0.0f);
    }

    template <typename Config>
    __device__
    void operator()(
        Config config
      , const mf32_span g
      , cuda::std::span<mf32_span> xi
      , cuda::std::span<mf32_span> V
      , cuda::std::span<mf32_span> V_bar
      , const cuda::std::span<float> ci
      , float 𝜆, float 𝜏1
      , cuda::std::span<float> max_delta_q
      )
    {
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        __shared__ typename BlockReduce::TempStorage temp;

        const auto k = int(V.size()) + 1;

        auto m   = g.mapping();
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto Δq  = 0.0f;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto q  = Matrix<float, Dynamic, 1, ColMajor, max_k, 1>{k};
                auto v  = Matrix<float, Dynamic, 1, ColMajor, max_k, 1>{k};
                auto vk = 1.0f;
                auto vi = 0.0f;
                for (auto i = 0; k > i; ++i) {
                    auto div = 0.0f;

                    if (0 < n)                div -= xi[i * 3][n - 1, row, col];
                    if (m.n_images() - 1 > n) div += xi[i * 3][n, row, col];

                    if (m.row_begin(col) < row)   div -= xi[i * 3 + 1][n, row - 1, col];
                    if (m.row_end(col) - 1 > row) div += xi[i * 3 + 1][n, row, col];

                    if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                        div -= xi[i * 3 + 2][n, row, col - 1];
                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                        div += xi[i * 3 + 2][n, row, col];

                    if (k - 1 > i) {
                        vi = V[i][n, row, col];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }

                    v[i] = vi;
                    q[i] = vi + 𝜏1 * (div - 𝜆 * sqr(g[n, row, col] - ci[i]));
                }
                project_simplex(q);
                for (auto i = 0; k - 1 > i; ++i) {
                    V_bar[i][n, row, col] = 2 * q[i] - V[i][n, row, col];
                    V[i][n, row, col] = q[i];
                }
                Δq = (q - v).squaredNorm();
            }
        }
        auto block_max = BlockReduce(temp).Reduce(Δq, cuda::maximum<>{});
        if (0 == cuda::gpu_thread.rank(cuda::block, config)) {
            cuda::atomic_ref<float, cuda::thread_scope_device>{max_delta_q[0]}
                .fetch_max(block_max, cuda::std::memory_order_relaxed);
        }
    }
};

struct update_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , mf32_span g
      , const cuda::std::span<mf32_span> V
      , const cuda::std::span<float> ci
      )
    {
        const auto k = int(V.size()) + 1;

        auto m   = g.mapping();
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
#if 1
                auto v  = 0.0f;
                auto vk = 1.0f;
                auto vi = 0.0f;
                auto max_label = 0;
                for (auto i = 0; k > i; ++i) {
                    if (k - 1 > i) {
                        vi = V[i][n, row, col];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    if (vi > v) {
                        v = vi;
                        max_label = i;
                    }
                }
                g[n, row, col] = ci[max_label];
#else
                auto v  = 0.0f;
                auto vk = 1.0f;
                auto vi = 0.0f;
                for (auto i = 0; k > i; ++i) {
                    if (k - 1 > i) {
                        vi = V[i][n, row, col];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    v += vi * ci[i];
                }
                g[n, row, col] = v;
#endif
            }
        }
    }
};

void potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , std::span<float> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  )
{
    auto mapping = g.mapping();

    static auto constexpr d = 3;
    const auto k  = int(ci.size());

    if (max_k < k) {
        throw std::runtime_error("Two many components");
    }

    auto zero_view = [&](mf32_span d) {
        cuda::fill_bytes(
            ctx.stream()
          , cuda::std::span{
                d.data_handle()
              , d.data_handle()  + mapping.required_span_size()
              }
          , 0);
    };

    auto Ξ     = std::vector<DeviceField<float>>();
    auto V     = std::vector<DeviceField<float>>();
    auto V_bar = std::vector<DeviceField<float>>();
    for (auto i = 0; k * d > i; ++i) {
        Ξ.emplace_back(mapping, ctx.stream(), ctx.mr());
        zero_view(Ξ.back().view());
    }
    for (auto i = 0; k - 1 > i; ++i) {
        V    .emplace_back(mapping, ctx.stream(), ctx.mr());
        V_bar.emplace_back(mapping, ctx.stream(), ctx.mr());
    }

    auto V_views = cuda::device_buffer<mf32_span>{
        ctx.stream(), ctx.mr()
      , rv::transform(V, [](auto &&v) { return v.view(); }) | rs::to_vector
      };
    auto V_bar_views = cuda::device_buffer<mf32_span>{
        ctx.stream(), ctx.mr()
      , rv::transform(V_bar, [](auto &&v) { return v.view(); }) | rs::to_vector
      };
    auto Ξ_views = cuda::device_buffer<mf32_span>{
        ctx.stream(), ctx.mr()
      , rv::transform(Ξ, [](auto &&v) { return v.view(); }) | rs::to_vector
      };
    auto d_ci = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), ci};

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

    cuda::launch(ctx.stream(), config, init_v_kernel{}, g.view(), V_views, V_bar_views, d_ci);
    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        cuda::launch(ctx.stream(), config, grad_kernel{}, Ξ_views, V_bar_views, 𝜏2);

        cuda::fill_bytes(ctx.stream(), d_maxΔq, 0);
        cuda::launch(ctx.stream(), config, div_kernel{}, g.view(), Ξ_views, V_views, V_bar_views, d_ci, 𝜆, 𝜏1, d_maxΔq);

        auto maxΔq = 0.0f;
        cuda::copy_bytes(ctx.stream(), d_maxΔq, cuda::std::span{&maxΔq, 1});
        ctx.stream().sync();

        fmt::print("{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
        // fmt::print(f.get(), "{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
    }
    cuda::launch(ctx.stream(), config, update_kernel{}, g.view(), V_views, d_ci);
}
