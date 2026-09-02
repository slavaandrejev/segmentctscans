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
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/utility>

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
    template <typename Config, typename ...Xn, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , const float *g
      , std::array<float *, sizeof...(Xn) - 1> V
      , std::array<float *, sizeof...(Xn) - 1> V_bar
      , const hana::tuple<Xn...> ci
      )
    {
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto offset = m(n, row, col);
                auto v = g[offset];
                auto [min_𝛿, cluster] = hana::fold(
                    hana::make_range(0_c, hana::size_c<sizeof...(Xn)>)
                  , std::tuple{std::numeric_limits<decltype(v)>::max(), 0}
                  , [&](auto acc, auto k) {
                        auto 𝛿 = std::abs(ci[k] - v);
                        return 𝛿 < std::get<0>(acc) ? std::tuple{𝛿, int(k)} : acc;
                  });
                for (auto i = 0; int(V.size()) > i; ++i) {
                    V_bar[i][offset] = V[i][offset] = i == cluster ? 1.0f : 0.0f;
                }
            }
        }
    }
};

struct grad_kernel {
    template <typename Config, typename Extents, size_t Km1>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , std::array<float *, 3 * Km1> xi
      , std::array<float *, Km1> V_bar
      , float 𝜏2
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;

        static constexpr auto K = Km1 + 1;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
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

                auto vk = 1.0f, vk_n = 1.0f, vk_r = 1.0f, vk_c = 1.0f;
                auto vi = 0.0f, vi_n = 0.0f, vi_r = 0.0f, vi_c = 0.0f;

                auto xi_k = std::array<float, 3>{};
                auto xi_i = xi_k;

                auto q = std::array<float, K * 3>{};
                STATIC_FOR(i)
                    if constexpr (hana::llong_c<K> - 1_c > i) {
                        vi = V_bar[i][offset];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    if (m.n_images() - 1 > n) {
                        if constexpr (hana::llong_c<K> - 1 > i) {
                            vi_n = V_bar[i][next_img_ofs];
                            vk_n -= vi_n;
                        } else {
                            vi_n = vk_n;
                        }
                        q[3_c * i] = vi_n - vi;
                    }

                    if (m.row_end(col) - 1 > row) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            vi_r = V_bar[i][next_row_ofs];
                            vk_r -= vi_r;
                        } else {
                            vi_r = vk_r;
                        }
                        q[3_c * i + 1_c] = vi_r - vi;
                    }

                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            vi_c = V_bar[i][next_col_ofs];
                            vk_c -= vi_c;
                        } else {
                            vi_c = vk_c;
                        }
                        q[3_c * i + 2_c] = vi_c - vi;
                    }

                    if constexpr (hana::llong_c<K> - 1_c > i) {
                        xi_i = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...axis) {
                            return std::array<float, 3>{{xi[i * 3_c + axis][offset]...}};
                        });
                        hana::for_each(hana::make_range(0_c, 3_c), [&](auto axis) {
                            xi_k[axis] -= xi_i[axis];
                        });
                    } else {
                        xi_i = xi_k;
                    }

                    q[3_c * i] *= 𝜏2;
                    q[3_c * i] += xi_i[0_c];

                    q[3_c * i + 1_c] *= 𝜏2;
                    q[3_c * i + 1_c] += xi_i[1_c];

                    q[3_c * i + 2_c] *= 𝜏2;
                    q[3_c * i + 2_c] += xi_i[2_c];
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
                hana::for_each(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto i) {
                    xi[i * 3_c      ][offset] = q[3_c * i];
                    xi[i * 3_c + 1_c][offset] = q[3_c * i + 1_c];
                    xi[i * 3_c + 2_c][offset] = q[3_c * i + 2_c];
                });
            }
        }
    }
};

struct div_kernel {
    // taken from https://bertdobbelaere.github.io/sorting_networks.html
    static constexpr auto sorting_networks = hana::make_map(
        hana::make_pair(
            2_c
          , hana::make_tuple(
                hana::make_pair(0_c, 1_c)
              )
          )
      , hana::make_pair(
            3_c
          , hana::make_tuple(
                hana::make_pair(0_c, 2_c)
              , hana::make_pair(0_c, 1_c)
              , hana::make_pair(1_c, 2_c)
              )
          )
      , hana::make_pair(
            4_c
          , hana::make_tuple(
                hana::make_pair(0_c, 2_c)
              , hana::make_pair(1_c, 3_c)
              , hana::make_pair(0_c, 1_c)
              , hana::make_pair(2_c, 3_c)
              , hana::make_pair(1_c, 2_c)
              )
          )
      , hana::make_pair(
            5_c
          , hana::make_tuple(
                hana::make_pair(0_c, 3_c)
              , hana::make_pair(1_c, 4_c)
              , hana::make_pair(0_c, 2_c)
              , hana::make_pair(1_c, 3_c)
              , hana::make_pair(0_c, 1_c)
              , hana::make_pair(2_c, 4_c)
              , hana::make_pair(1_c, 2_c)
              , hana::make_pair(3_c, 4_c)
              , hana::make_pair(2_c, 3_c)
              )
          )
      );

    template <typename ...Xn>
    static __forceinline__
    __device__ auto project_simplex(hana::tuple<Xn...> q)
    {
        static auto constexpr K = sizeof...(Xn);

        auto u = q;

        constexpr auto key = hana::llong_c<K>;
        constexpr auto has_network = hana::contains(sorting_networks, key);

        auto ce = [&](auto a, auto b) {
            const float x = u[a];
            const float y = u[b];

            u[a] = fmaxf(x, y);
            u[b] = fminf(x, y);
        };

        if constexpr (has_network) {
            constexpr auto sort_nw = hana::at_key(sorting_networks, key);
            hana::for_each(sort_nw, [&](auto p) {
                ce(hana::first(p), hana::second(p));
            });
        } else {
            hana::for_each(hana::make_range(1_c, hana::llong_c<K>), [&](auto i) {
                hana::for_each(hana::make_range(0_c, i), [&](auto jj) {
                    constexpr auto j = i - 1_c - jj;
                    ce(j, j + 1_c);
                });
            });
        }

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

    template <typename Config, typename ...Xn, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , const float *g
      , std::array<float *, 3 * (sizeof...(Xn) - 1)> xi
      , std::array<float *, sizeof...(Xn) - 1> V
      , std::array<float *, sizeof...(Xn) - 1> V_bar
      , const hana::tuple<Xn...> ci
      , float 𝜆, float 𝜏1
      , cuda::std::span<float> max_delta_q
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        static auto constexpr K = sizeof...(Xn);

        __shared__ typename BlockReduce::TempStorage temp;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto Δq  = 0.0f;
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

                auto q  = hana::replicate<hana::tuple_tag>(0.0f, hana::size_c<K>);
                auto v  = q;

                auto vk = 1.0f; // the last V component we are going to calculate
                auto vi = 0.0f;

                // the last xi component we are goint to compute
                auto xi_k = hana::replicate<hana::tuple_tag>(0.0f, 3_c);
                auto xi_k_n = 0.0f, xi_k_r = 0.0f, xi_k_c = 0.0f;
                auto xi_i = xi_k;
                auto xi_i_n = 0.0f, xi_i_r = 0.0f, xi_i_c = 0.0f;

                STATIC_FOR(i)
                    auto div = 0.0f;

                    if constexpr (hana::llong_c<K> - 1_c > i) {
                        xi_i = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...axis) {
                            return hana::make_tuple(xi[i * 3_c + axis][offset]...);
                        });
                        hana::for_each(hana::make_range(0_c, 3_c), [&](auto axis) {
                            xi_k[axis] -= xi_i[axis];
                        });
                    } else {
                        xi_i = xi_k;
                    }

                    if (0 < n) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            xi_i_n = xi[i * 3_c][prev_img_ofs];
                            xi_k_n -= xi_i_n;
                        } else {
                            xi_i_n = xi_k_n;
                        }
                        div -= xi_i_n;
                    }
                    if (m.n_images() - 1 > n) div += xi_i[0_c];

                    if (m.row_begin(col) < row) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            xi_i_r = xi[i * 3_c + 1_c][prev_row_ofs];
                            xi_k_r -= xi_i_r;
                        } else {
                            xi_i_r = xi_k_r;
                        }
                        div -= xi_i_r;
                    }
                    if (m.row_end(col) - 1 > row) div += xi_i[1_c];

                    if (m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1)) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            xi_i_c = xi[i * 3_c + 2_c][prev_col_ofs];
                            xi_k_c -= xi_i_c;
                        } else {
                            xi_i_c = xi_k_c;
                        }
                        div -= xi_i_c;
                    }
                    if (m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1))
                        div += xi_i[2_c];

                    if constexpr (hana::llong_c<K> - 1_c> i) {
                        vi = V[i][offset];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }

                    v[i] = vi;
                    q[i] = vi + 𝜏1 * (div - 𝜆 * sqr(g[offset] - ci[i]));
                STATIC_FOR_END(i, 0, K);
                q = project_simplex(q);
                hana::for_each(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto i) {
                    V_bar[i][offset] = 2 * q[i] - v[i];
                    V[i][offset] = q[i];
                });
                Δq = hana::fold(
                    hana::make_range(0_c, hana::llong_c<K>)
                  , 0.0f
                  , [&](auto acc, auto k) {
                        return acc + sqr(q[k] - v[k]);
                    }
                  );
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
    template <typename Config, typename Extents, size_t K>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , float *g
      , std::array<float *, K> V
      , const cuda::std::span<float> ci
      )
    {
        const auto k = int(V.size()) + 1;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                auto offset = m(n, row, col);
#if 1
                auto v  = 0.0f;
                auto vk = 1.0f;
                auto vi = 0.0f;
                auto max_label = 0;
                for (auto i = 0; k > i; ++i) {
                    if (k - 1 > i) {
                        vi = V[i][offset];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    if (vi > v) {
                        v = vi;
                        max_label = i;
                    }
                }
                g[offset] = ci[max_label];
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
    // non-owning mapping object with pointers to the device mapping tables, can
    // be passed to a kernel by value
    auto mapping = g.mapping();

    static auto constexpr d = 3;

    // Vector fields for each pixel and each but one label. They sum up to zero,
    // so need K - 1 components.
    auto Ξ = hana::unpack(hana::make_range(0_c, hana::llong_c<d * (K - 1)>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, d * (K - 1)>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), mapping.required_span_size(), cuda::no_init})...
        }};
    });
    // V and V_bar sum up to one, so need K - 1 components.
    auto V = hana::unpack(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, K - 1>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), mapping.required_span_size(), cuda::no_init})...
        }};
    });
    auto V_bar = hana::unpack(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, K - 1>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), mapping.required_span_size(), cuda::no_init})...
        }};
    });
    for (auto &&𝜉 : Ξ) {
        cuda::fill_bytes(ctx.stream(), 𝜉, 0.0f);
    }
    for (auto &&v : V) {
        cuda::fill_bytes(ctx.stream(), v, 0.0f);
    }
    for (auto &&v : V_bar) {
        cuda::fill_bytes(ctx.stream(), v, 0.0f);
    }

    auto Ξ_ptrs = hana::unpack(Ξ, [](auto&... buffer) {
        return std::array<float*, d * (K - 1)>{{buffer.data()...}};
    });
    auto V_ptrs = hana::unpack(V, [](auto&... buffer) {
        return std::array<float*, K - 1>{{buffer.data()...}};
    });
    auto V_bar_ptrs = hana::unpack(V_bar, [](auto&... buffer) {
        return std::array<float*, K - 1>{{buffer.data()...}};
    });


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

    cuda::launch(ctx.stream(), config, init_v_kernel{}, mapping, g.data(), V_ptrs, V_bar_ptrs, ci_tuple);
    for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
        cuda::launch(ctx.stream(), config, grad_kernel{}, mapping, Ξ_ptrs, V_bar_ptrs, 𝜏2);

        cuda::fill_bytes(ctx.stream(), d_maxΔq, 0);
        cuda::launch(ctx.stream(), config, div_kernel{}, mapping, g.data(), Ξ_ptrs, V_ptrs, V_bar_ptrs, ci_tuple, 𝜆, 𝜏1, d_maxΔq);

        auto maxΔq = 0.0f;
        cuda::copy_bytes(ctx.stream(), d_maxΔq, cuda::std::span{&maxΔq, 1});
        ctx.stream().sync();

        fmt::print("{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
        // fmt::print(f.get(), "{} {:.6g}\n", n_iter, std::sqrt(maxΔq));
    }
    auto d_ci = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), ci};
    cuda::launch(ctx.stream(), config, update_kernel{}, mapping, g.data(), V_ptrs, d_ci);
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
