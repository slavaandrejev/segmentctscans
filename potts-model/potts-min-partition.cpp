#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <boost/hana.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/ext/std/array.hpp>

#include <cub/block/block_reduce.cuh>
#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/cmath>

#include <fmt/printf.h>

#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>

#include <sqr.h>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include "potts-min-partition.h"

namespace hana = boost::hana;
namespace rs   = ranges;
namespace rv   = rs::views;

using namespace hana::literals;

static constexpr auto threads_per_block = 256;
static constexpr auto max_k = 6;

template <long long Begin, long long End>
constexpr auto make_llong_range() {
    return []<long long... I>(std::integer_sequence<long long, I...>) {
        return std::integer_sequence<long long, (I + Begin)...>{};
    }(std::make_integer_sequence<long long, End - Begin>{});
}

#define STATIC_FOR(var)                                                         \
    [&]<long long... var##_Is>(std::integer_sequence<long long, var##_Is...>) { \
        ([&](auto var) {

#define STATIC_FOR_END(var, Begin, End)   \
        }(hana::llong_c<var##_Is>), ...); \
    }(make_llong_range<Begin, End>())

struct init_v_kernel {
    template <typename Config, typename ...Xn>
    __device__
    void operator()(
        Config config
      , const mf32_span g
      , cuda::std::span<mf32_span> V
      , cuda::std::span<mf32_span> V_bar
      , const hana::tuple<Xn...> ci
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
                auto [min_𝛿, cluster] = hana::fold(
                    hana::make_range(0_c, hana::size_c<sizeof...(Xn)>)
                  , std::tuple{std::numeric_limits<decltype(v)>::max(), 0}
                  , [&](auto acc, auto k) {
                        auto 𝛿 = std::abs(ci[k] - v);
                        return 𝛿 < std::get<0>(acc) ? std::tuple{𝛿, int(k)} : acc;
                  });
                for (auto i = 0; int(V.size()) > i; ++i) {
                    V_bar[i][n, row, col] = V[i][n, row, col] = i == cluster ? 1.0f : 0.0f;
                }
            }
        }
    }
};

template <size_t K>
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
        auto m   = xi[0].mapping();
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto vk = 1.0f, vk_n = 1.0f, vk_r = 1.0f, vk_c = 1.0f;
                auto vi = 0.0f, vi_n = 0.0f, vi_r = 0.0f, vi_c = 0.0f;
                auto q = std::array<float, K * 3>{};
                STATIC_FOR(i)
                    if constexpr (hana::llong_c<K> - 1_c > i) {
                        vi = V_bar[i][n, row, col];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    if (m.n_images() - 1 > n) {
                        if constexpr (hana::llong_c<K> - 1 > i) {
                            vi_n = V_bar[i][n + 1, row, col];
                            vk_n -= vi_n;
                        } else {
                            vi_n = vk_n;
                        }
                        q[3_c * i] = vi_n - vi;
                    }

                    if (m.row_end(col) - 1 > row) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            vi_r = V_bar[i][n, row + 1, col];
                            vk_r -= vi_r;
                        } else {
                            vi_r = vk_r;
                        }
                        q[3_c * i + 1_c] = vi_r - vi;
                    }

                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            vi_c = V_bar[i][n, row, col + 1];
                            vk_c -= vi_c;
                        } else {
                            vi_c = vk_c;
                        }
                        q[3_c * i + 2_c] = vi_c - vi;
                    }

                    q[3_c * i] *= 𝜏2;
                    q[3_c * i] += xi[i * 3_c][n, row, col];

                    q[3_c * i + 1_c] *= 𝜏2;
                    q[3_c * i + 1_c] += xi[i * 3_c + 1_c][n, row, col];

                    q[3_c * i + 2_c] *= 𝜏2;
                    q[3_c * i + 2_c] += xi[i * 3_c + 2_c][n, row, col];
                STATIC_FOR_END(i, 0, K);
                auto qi1i2 = std::array<std::array<float, 3>, K * (K - 1) / 2>{};
                for (auto sweep = 0; 20 > sweep; ++sweep) {
                    auto max_𝛿 = std::numeric_limits<float>::lowest();
                    STATIC_FOR(i1)
                        STATIC_FOR(i2)
                            auto constexpr qi1i2_idx = (2_c * hana::llong_c<K> - 1_c - i1) * i1 / 2_c + i2 - i1 - 1_c;
                            auto qh = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...n) {
                                return std::array<float, 3>{{
                                    (q[3_c * i2 + n] - q[3_c * i1 + n] + qi1i2[qi1i2_idx][n])...
                                }};
                            });
                            auto l = cuda::std::sqrt(hana::fold(qh, 0.0f, [](auto acc, auto x) { return acc + x * x; }));
                            auto qhh = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...n) {
                                return std::array<float, 3>{{
                                    (l > 1 ? ((l - 1.0f) * qh[n] / l) : 0.0f)...
                                }};
                            });
                            auto 𝛿 = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...n) {
                                return std::array<float, 3>{{
                                    0.5f * (qhh[n] - qi1i2[qi1i2_idx][n])...
                                }};
                            });
                            auto 𝛿_norm = cuda::std::sqrt(hana::fold(𝛿, 0.0f, [](auto acc, auto x) { return acc + x * x; }));
                            max_𝛿 = cuda::std::max(𝛿_norm, max_𝛿);
                            hana::for_each(hana::make_range(0_c, 3_c), [&](auto n) {
                                q[3_c * i1 + n] += 𝛿[n];
                                q[3_c * i2 + n] -= 𝛿[n];
                            });
                            qi1i2[qi1i2_idx] = qhh;
                        STATIC_FOR_END(i2, hana::value(i1 + 1_c), K);
                    STATIC_FOR_END(i1, 0, K - 1);
                    if (1e-6 > max_𝛿) {
                        break;
                    }
                }
                hana::for_each(hana::make_range(0_c, hana::llong_c<K>), [&](auto i) {
                    xi[i * 3_c      ][n, row, col] = q[3_c * i];
                    xi[i * 3_c + 1_c][n, row, col] = q[3_c * i + 1_c];
                    xi[i * 3_c + 2_c][n, row, col] = q[3_c * i + 2_c];
                });
            }
        }
    }
};

struct div_kernel {
    template <typename ...Xn>
    static __forceinline__
    __device__ auto project_simplex(hana::tuple<Xn...> q)
    {
        static auto constexpr K = sizeof...(Xn);

        auto u = q;

        // insertion sort
        hana::for_each(hana::make_range(1_c, hana::llong_c<K>), [&](auto i) {
            auto key = u[i];
            auto done = false;
            hana::for_each(hana::make_range(0_c, i), [&](auto jj) {
                auto j = i - 1_c - jj;
                if (!done) {
                    if (u[j] < key) {
                        u[j + 1_c] = u[j];
                    } else {
                        done = true;
                        u[j + 1_c] = key;
                    }
                }
            });
            if (!done) { u[0_c] = key; }
        });

        auto sum   = 0.0f;
        auto 𝜌_sum = 0.0f;
        auto 𝜌     = 0;

        hana::for_each(hana::make_range(0_c, hana::llong_c<K>), [&](auto i) {
            sum += u[i];
            auto j = i + 1_c;
            const auto theta = (sum - 1.0f) / hana::value(j);
            if (u[i] > theta) {
                𝜌 = j;
                𝜌_sum = sum;
            }
        });

        const float theta = (𝜌_sum - 1.0f) / 𝜌;

        hana::for_each(hana::make_range(0_c, hana::llong_c<K>), [&](auto i) {
            q[i] = cuda::std::max(q[i] - theta, 0.0f);
        });

        return q;
    }

    template <typename Config, typename ...Xn>
    __device__
    void operator()(
        Config config
      , const mf32_span g
      , cuda::std::span<mf32_span> xi
      , cuda::std::span<mf32_span> V
      , cuda::std::span<mf32_span> V_bar
      , const hana::tuple<Xn...> ci
      , float 𝜆, float 𝜏1
      , cuda::std::span<float> max_delta_q
      )
    {
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        static auto constexpr K = sizeof...(Xn);

        __shared__ typename BlockReduce::TempStorage temp;

        auto m   = g.mapping();
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto Δq  = 0.0f;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto q  = hana::replicate<hana::tuple_tag>(0.0f, hana::size_c<K>);
                auto v  = hana::replicate<hana::tuple_tag>(0.0f, hana::size_c<K>);
                auto vk = 1.0f;
                auto vi = 0.0f;
                STATIC_FOR(i)
                    auto div = 0.0f;

                    if (0 < n)                div -= xi[i * 3_c][n - 1, row, col];
                    if (m.n_images() - 1 > n) div += xi[i * 3_c][n, row, col];

                    if (m.row_begin(col) < row)   div -= xi[i * 3_c + 1_c][n, row - 1, col];
                    if (m.row_end(col) - 1 > row) div += xi[i * 3_c + 1_c][n, row, col];

                    if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                        div -= xi[i * 3_c + 2_c][n, row, col - 1];
                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                        div += xi[i * 3_c + 2_c][n, row, col];

                    if constexpr (hana::llong_c<K> - 1_c> i) {
                        vi = V[i][n, row, col];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }

                    v[i] = vi;
                    q[i] = vi + 𝜏1 * (div - 𝜆 * sqr(g[n, row, col] - ci[i]));
                STATIC_FOR_END(i, 0, K);
                q = project_simplex(q);
                hana::for_each(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto i) {
                    V_bar[i][n, row, col] = 2 * q[i] - v[i];
                    V[i][n, row, col] = q[i];
                });
                Δq = cuda::std::sqrt(hana::fold(
                    hana::make_range(0_c, hana::llong_c<K>)
                  , 0.0f
                  , [&](auto acc, auto k) {
                        return acc + sqr(q[k] - v[k]);
                    }
                  ));
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

template <size_t K>
void potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , std::array<float, K> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  )
{
    auto mapping = g.mapping();

    static auto constexpr d = 3;

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
    for (auto i = 0; K * d > i; ++i) {
        Ξ.emplace_back(mapping, ctx.stream(), ctx.mr());
        zero_view(Ξ.back().view());
    }
    for (auto i = 0; K - 1 > i; ++i) {
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

    auto ci_tuple = hana::unpack(ci, [&](auto ...c) {
        return hana::make_tuple(c...);
    });

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

    cuda::launch(ctx.stream(), config, init_v_kernel{}, g.view(), V_views, V_bar_views, ci_tuple);
    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        cuda::launch(ctx.stream(), config, grad_kernel<K>{}, Ξ_views, V_bar_views, 𝜏2);

        cuda::fill_bytes(ctx.stream(), d_maxΔq, 0);
        cuda::launch(ctx.stream(), config, div_kernel{}, g.view(), Ξ_views, V_views, V_bar_views, ci_tuple, 𝜆, 𝜏1, d_maxΔq);

        auto maxΔq = 0.0f;
        cuda::copy_bytes(ctx.stream(), d_maxΔq, cuda::std::span{&maxΔq, 1});
        ctx.stream().sync();

        fmt::print("{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
        // fmt::print(f.get(), "{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
    }
    auto d_ci = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), ci};
    cuda::launch(ctx.stream(), config, update_kernel{}, g.view(), V_views, d_ci);
}

// template <std::size_t... Ks>
// constexpr auto instantiate_potts(std::index_sequence<Ks...>)
// {
//     return std::tuple{&potts_min_partition<Ks + 2>...};
// }

// [[maybe_unused]] constinit auto potts_instances =
//     instantiate_potts(std::make_index_sequence<max_k - 1>{});

template
void potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , std::array<float, 3> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  );
