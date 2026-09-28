#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <vector>

#include <boost/hana/at_key.hpp>
#include <boost/hana/div.hpp>
#include <boost/hana/ext/std/array.hpp>
#include <boost/hana/ext/std/integral_constant.hpp>
#include <boost/hana/fold.hpp>
#include <boost/hana/for_each.hpp>
#include <boost/hana/greater.hpp>
#include <boost/hana/integral_constant.hpp>
#include <boost/hana/map.hpp>
#include <boost/hana/minus.hpp>
#include <boost/hana/mult.hpp>
#include <boost/hana/plus.hpp>
#include <boost/hana/range.hpp>
#include <boost/hana/replicate.hpp>
#include <boost/hana/tuple.hpp>
#include <boost/hana/unpack.hpp>

#include <cub/block/block_reduce.cuh>

#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/utility>

#include <cuda/std/array>

#include <fmt/color.h>
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

struct init_v_kernel {
    template <typename Config, typename ...Xn, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> input_mapping
      , layout_cylinder::mapping<Extents> slab_mapping
      , const float *g
      , cuda::std::array<float *, sizeof...(Xn) - 1> V
      , cuda::std::array<float *, sizeof...(Xn) - 1> V_bar
      , const hana::tuple<Xn...> ci
      , int slab_start
      , int slab_end
      )
    {
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = slab_mapping.col_begin() + b.z;
        auto row = b.y;
        if (slab_mapping.row_begin(col) <= row && row < slab_mapping.row_end(col)) {
            auto n = slab_start + cuda::gpu_thread.index(cuda::grid, config).x;

            if (slab_end > n) {
                auto input_offset = input_mapping(n, row, col);
                auto work_offset  = slab_mapping(n - slab_start, row, col);
                auto v = g[input_offset];
                auto [min_𝛿, cluster] = hana::fold(
                    hana::make_range(0_c, hana::size_c<sizeof...(Xn)>)
                  , std::tuple{std::numeric_limits<decltype(v)>::max(), 0}
                  , [&](auto acc, auto k) {
                        auto 𝛿 = std::abs(ci[k] - v);
                        return 𝛿 < std::get<0>(acc) ? std::tuple{𝛿, int(k)} : acc;
                  });
                for (auto i = 0; int(V.size()) > i; ++i) {
                    V_bar[i][work_offset] = V[i][work_offset] = i == cluster ? 1.0f : 0.0f;
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
      , layout_cylinder::mapping<Extents> slab_mapping
      , cuda::std::array<float *, 3 * Km1> xi
      , cuda::std::array<float *, Km1> V_bar
      , float 𝜏2
      , int slab_start
      , int slab_end
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;

        static constexpr auto K = Km1 + 1;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = slab_mapping.col_begin() + b.z;
        auto row = b.y;
        if (slab_mapping.row_begin(col) <= row && row < slab_mapping.row_end(col)) {
            auto n = slab_start + cuda::gpu_thread.index(cuda::grid, config).x;

            if (slab_end > n) {
                auto work_offset  = slab_mapping(n - slab_start, row, col);
                auto next_img_ofs = index_type{};
                auto next_row_ofs = index_type{};
                auto next_col_ofs = index_type{};

                if (slab_end - 1 > n) {
                    next_img_ofs = slab_mapping(n + 1 - slab_start, row, col);
                }
                if (slab_mapping.row_end(col) - 1 > row) {
                    next_row_ofs = slab_mapping(n - slab_start, row + 1, col);
                }
                if (slab_mapping.col_end() - 1 > col && slab_mapping.row_begin(col + 1) <= row && row < slab_mapping.row_end(col + 1)) {
                    next_col_ofs = slab_mapping(n - slab_start, row, col + 1);
                }

                auto vk = 1.0f, vk_n = 1.0f, vk_r = 1.0f, vk_c = 1.0f;
                auto vi = 0.0f, vi_n = 0.0f, vi_r = 0.0f, vi_c = 0.0f;

                auto xi_k = hana::replicate<hana::tuple_tag>(0.0f, 3_c);
                auto xi_i = xi_k;

                auto q = hana::replicate<hana::tuple_tag>(0.0f, hana::llong_c<K> * 3_c);
                auto qi1i2 = hana::replicate<hana::tuple_tag>(xi_k, hana::llong_c<K * (K - 1) / 2>);
                cuda::static_for<K>([&](auto I) {
                    static constexpr auto i = hana::llong_c<decltype(I)::value>;

                    if constexpr (hana::llong_c<K> - 1_c > i) {
                        vi = V_bar[i][work_offset];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    if (slab_end - 1 > n) {
                        if constexpr (hana::llong_c<K> - 1 > i) {
                            vi_n = V_bar[i][next_img_ofs];
                            vk_n -= vi_n;
                        } else {
                            vi_n = vk_n;
                        }
                        q[3_c * i] = vi_n - vi;
                    }

                    if (slab_mapping.row_end(col) - 1 > row) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            vi_r = V_bar[i][next_row_ofs];
                            vk_r -= vi_r;
                        } else {
                            vi_r = vk_r;
                        }
                        q[3_c * i + 1_c] = vi_r - vi;
                    }

                    if (slab_mapping.col_end() - 1 > col && slab_mapping.row_begin(col + 1) <= row && row < slab_mapping.row_end(col + 1)) {
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
                            return hana::make_tuple(xi[i * 3_c + axis][work_offset]...);
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
                });
                for (auto sweep = 0; 20 > sweep; ++sweep) {
                    auto max_𝛿 = std::numeric_limits<float>::lowest();
                    cuda::static_for<0, K - 1>([&](auto I1) {
                        static constexpr auto i1 = hana::llong_c<decltype(I1)::value>;
                        cuda::static_for<I1 + 1, K>([&](auto I2) {
                            static constexpr auto i2 = hana::llong_c<decltype(I2)::value>;

                            auto constexpr qi1i2_idx = (2_c * hana::llong_c<K> - 1_c - i1) * i1 / 2_c + i2 - i1 - 1_c;
                            auto qh = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...n) {
                                return hana::make_tuple(
                                    (q[3_c * i2 + n] - q[3_c * i1 + n] + qi1i2[qi1i2_idx][n])...
                                );
                            });
                            auto l = cuda::std::sqrt(hana::fold(qh, 0.0f, [](auto acc, auto x) { return acc + x * x; }));
                            auto qhh = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...n) {
                                return hana::make_tuple(
                                    (l > 1 ? ((l - 1.0f) * qh[n] / l) : 0.0f)...
                                );
                            });
                            auto 𝛿 = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...n) {
                                return hana::make_tuple(
                                    0.5f * (qhh[n] - qi1i2[qi1i2_idx][n])...
                                );
                            });
                            auto 𝛿_norm = cuda::std::sqrt(hana::fold(𝛿, 0.0f, [](auto acc, auto x) { return acc + x * x; }));
                            max_𝛿 = cuda::std::max(𝛿_norm, max_𝛿);
                            hana::for_each(hana::make_range(0_c, 3_c), [&](auto n) {
                                q[3_c * i1 + n] += 𝛿[n];
                                q[3_c * i2 + n] -= 𝛿[n];
                            });
                            qi1i2[qi1i2_idx] = qhh;
                        });
                    });
                    if (1e-6 > max_𝛿) {
                        break;
                    }
                }
                hana::for_each(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto i) {
                    xi[i * 3_c      ][work_offset] = q[3_c * i];
                    xi[i * 3_c + 1_c][work_offset] = q[3_c * i + 1_c];
                    xi[i * 3_c + 2_c][work_offset] = q[3_c * i + 2_c];
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

        // use sorting networks for a few known K, fallback to insertion sort
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
      , layout_cylinder::mapping<Extents> input_mapping
      , layout_cylinder::mapping<Extents> slab_mapping
      , const float *g
      , cuda::std::array<float *, 3 * (sizeof...(Xn) - 1)> xi
      , cuda::std::array<float *, sizeof...(Xn) - 1> V
      , cuda::std::array<float *, sizeof...(Xn) - 1> V_bar
      , const hana::tuple<Xn...> ci
      , float 𝜆, float 𝜏1
      , int slab_start
      , int slab_end
      , cuda::std::span<float> max_delta_q
      )
    {
        using index_type = layout_cylinder::mapping<Extents>::index_type;
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;

        static auto constexpr K = sizeof...(Xn);

        __shared__ typename BlockReduce::TempStorage temp;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = slab_mapping.col_begin() + b.z;
        auto row = b.y;
        auto Δq  = 0.0f;
        if (slab_mapping.row_begin(col) <= row && row < slab_mapping.row_end(col)) {
            auto n = slab_start + cuda::gpu_thread.index(cuda::grid, config).x;

            if (slab_end > n) {
                auto input_offset = input_mapping(n, row, col);
                auto work_offset  = slab_mapping(n - slab_start, row, col);
                auto prev_img_ofs = index_type{};
                auto prev_row_ofs = index_type{};
                auto prev_col_ofs = index_type{};

                if (slab_start < n) {
                    prev_img_ofs = slab_mapping(n - 1 - slab_start, row, col);
                }
                if (slab_mapping.row_begin(col) < row) {
                    prev_row_ofs = slab_mapping(n - slab_start, row - 1, col) ;
                }
                if (slab_mapping.col_begin() < col && slab_mapping.row_begin(col - 1) <= row && row < slab_mapping.row_end(col - 1)) {
                    prev_col_ofs = slab_mapping(n - slab_start, row, col - 1);
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

                cuda::static_for<K>([&](auto I) {
                    static constexpr auto i = hana::llong_c<decltype(I)::value>;

                    auto div = 0.0f;

                    if constexpr (hana::llong_c<K> - 1_c > i) {
                        xi_i = hana::unpack(hana::make_range(0_c, 3_c), [&](auto ...axis) {
                            return hana::make_tuple(xi[i * 3_c + axis][work_offset]...);
                        });
                        hana::for_each(hana::make_range(0_c, 3_c), [&](auto axis) {
                            xi_k[axis] -= xi_i[axis];
                        });
                    } else {
                        xi_i = xi_k;
                    }

                    if (slab_start < n) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            xi_i_n = xi[i * 3_c][prev_img_ofs];
                            xi_k_n -= xi_i_n;
                        } else {
                            xi_i_n = xi_k_n;
                        }
                        div -= xi_i_n;
                    }
                    if (slab_end - 1 > n) div += xi_i[0_c];

                    if (slab_mapping.row_begin(col) < row) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            xi_i_r = xi[i * 3_c + 1_c][prev_row_ofs];
                            xi_k_r -= xi_i_r;
                        } else {
                            xi_i_r = xi_k_r;
                        }
                        div -= xi_i_r;
                    }
                    if (slab_mapping.row_end(col) - 1 > row) div += xi_i[1_c];

                    if (slab_mapping.col_begin() < col && slab_mapping.row_begin(col - 1) <= row && row < slab_mapping.row_end(col - 1)) {
                        if constexpr (hana::llong_c<K> - 1_c > i) {
                            xi_i_c = xi[i * 3_c + 2_c][prev_col_ofs];
                            xi_k_c -= xi_i_c;
                        } else {
                            xi_i_c = xi_k_c;
                        }
                        div -= xi_i_c;
                    }
                    if (slab_mapping.col_end() - 1 > col && slab_mapping.row_begin(col + 1) <= row && row < slab_mapping.row_end(col + 1))
                        div += xi_i[2_c];

                    if constexpr (hana::llong_c<K> - 1_c> i) {
                        vi = V[i][work_offset];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }

                    v[i] = vi;
                    q[i] = vi + 𝜏1 * (div - 𝜆 * sqr(g[input_offset] - ci[i]));
                });
                q = project_simplex(q);
                hana::for_each(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto i) {
                    V_bar[i][work_offset] = 2 * q[i] - v[i];
                    V[i][work_offset] = q[i];
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
      , layout_cylinder::mapping<Extents> output_mapping
      , layout_cylinder::mapping<Extents> slab_mapping
      , float *res
      , cuda::std::array<float *, K> V
      , const cuda::std::span<float> ci
      , int slab_start
      , int res_start
      , int res_end
      )
    {
        const auto k = int(V.size()) + 1;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = slab_mapping.col_begin() + b.z;
        auto row = b.y;
        if (slab_mapping.row_begin(col) <= row && row < slab_mapping.row_end(col)) {
            auto n = res_start + cuda::gpu_thread.index(cuda::grid, config).x;

            if (res_end > n) {
                auto output_offset = output_mapping(n, row, col);
                auto work_offset   = slab_mapping(n - slab_start, row, col);

                auto v  = 0.0f;
                auto vk = 1.0f;
                auto vi = 0.0f;
                auto max_label = 0;
                for (auto i = 0; k > i; ++i) {
                    if (k - 1 > i) {
                        vi = V[i][work_offset];
                        vk -= vi;
                    } else {
                        vi = vk;
                    }
                    if (vi > v) {
                        v = vi;
                        max_label = i;
                    }
                }
                res[output_offset] = ci[max_label];
            }
        }
    }
};

template <size_t K>
device_field_ptr<float> potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , const Field<float>::mapping_type &cpu_map
  , std::array<float, K> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  )
{
    // non-owning mapping object with pointers to the device mapping tables, can
    // be passed to a kernel by value
    auto mapping = g.mapping();

    auto res = device_field_ptr<float>{
        new DeviceField<float>{mapping, ctx.stream(), ctx.mr()}
      };

    static auto constexpr d = 3;

    const auto [available_memory, total] = device_memory(ctx);

    const auto n    = int(mapping.n_images());
    const auto halo = int(iters);
    const auto s    = mapping.required_span_size() / mapping.nstride();

    const auto bytes_per_slice = sizeof(float) * s * (d + 2) * (K - 1);

    const auto slab_w = int(Field<float>::prev_stride(available_memory / bytes_per_slice));
    const auto res_w  = slab_w - 2 * halo;

    if (n > slab_w && res_w <= 0) {
        fmt::print(
            fmt::fg(fmt::color::light_coral) | fmt::emphasis::bold
          , "Not enought memory. Reduce the mask radius or the number of iterations.\n"
          );
        return {};
    }

    auto slab_map_st    = layout_cylinder::mapping_storage(cpu_map, slab_w, Field<float>::get_stride(slab_w));
    auto slab_map       = slab_map_st.mapping();
    auto d_slab_map_tbl = slab_map.device_tables(ctx.stream(), ctx.mr());
    auto d_slab_map     = slab_map.with_tables(d_slab_map_tbl.csc());

    // Vector fields for each pixel and each but one label. They sum up to zero,
    // so need K - 1 components.
    auto Ξ = hana::unpack(hana::make_range(0_c, hana::llong_c<d * (K - 1)>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, d * (K - 1)>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), slab_map.required_span_size(), cuda::no_init})...
        }};
    });
    // V and V_bar sum up to one, so need K - 1 components.
    auto V = hana::unpack(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, K - 1>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), slab_map.required_span_size(), cuda::no_init})...
        }};
    });
    auto V_bar = hana::unpack(hana::make_range(0_c, hana::llong_c<K - 1>), [&](auto ...k) {
        return std::array<cuda::device_buffer<float>, K - 1>{{
            (void(k), cuda::device_buffer<float>{ctx.stream(), ctx.mr(), slab_map.required_span_size(), cuda::no_init})...
        }};
    });
    for (auto &&v : V) {
        cuda::fill_bytes(ctx.stream(), v, 0.0f);
    }
    for (auto &&v : V_bar) {
        cuda::fill_bytes(ctx.stream(), v, 0.0f);
    }

    auto Ξ_ptrs = hana::unpack(Ξ, [](auto&... buffer) {
        return cuda::std::array<float*, d * (K - 1)>{{buffer.data()...}};
    });
    auto V_ptrs = hana::unpack(V, [](auto&... buffer) {
        return cuda::std::array<float*, K - 1>{{buffer.data()...}};
    });
    auto V_bar_ptrs = hana::unpack(V_bar, [](auto&... buffer) {
        return cuda::std::array<float*, K - 1>{{buffer.data()...}};
    });


    auto ci_tuple = hana::unpack(ci, [&](auto ...c) {
        return hana::make_tuple(c...);
    });
    auto d_ci = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), ci};

    auto d_maxΔq = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), 1, cuda::no_init};

    const auto n_slabs = n <= slab_w ? int{1} : 1 + (n - slab_w + res_w - 1) / res_w;
    for (auto n_slab = 0; n_slabs > n_slab; ++n_slab) {
        const auto slab_start = n_slab * res_w;
        const auto slab_end   = std::min(int(mapping.n_images()), slab_start + slab_w);

        auto slab_cfg = cuda::make_config(
            cuda::grid_dims(dim3{
                ((unsigned int)(slab_end - slab_start) + (threads_per_block - 1)) / threads_per_block
              , mapping.extents().extent(1)
              , mapping.col_end() - mapping.col_begin()
              })
          , cuda::block_dims<threads_per_block>());

        // Process [slab_start, slab_end)
        cuda::launch(
            ctx.stream(), slab_cfg, init_v_kernel{}, mapping, d_slab_map
          , g.data(), V_ptrs, V_bar_ptrs, ci_tuple, slab_start, slab_end);
        for (auto &&𝜉 : Ξ) { cuda::fill_bytes(ctx.stream(), 𝜉, 0.0f); }
        for (auto n_iter = int{}; iters > n_iter; ++n_iter) {
            cuda::launch(ctx.stream(), slab_cfg, grad_kernel{}, d_slab_map, Ξ_ptrs, V_bar_ptrs, 𝜏2, slab_start, slab_end);

            cuda::fill_bytes(ctx.stream(), d_maxΔq, 0);
            cuda::launch(
                ctx.stream(), slab_cfg, div_kernel{}, mapping, d_slab_map
              , g.data(), Ξ_ptrs, V_ptrs, V_bar_ptrs, ci_tuple, 𝜆, 𝜏1
              , slab_start, slab_end, d_maxΔq);

            auto maxΔq = 0.0f;
            cuda::copy_bytes(ctx.stream(), d_maxΔq, cuda::std::span{&maxΔq, 1});
            ctx.stream().sync();

            fmt::print("Slab = {:4}, iteration = {:4}, Δq = {:.6g}   \r", n_slab, n_iter, std::sqrt(maxΔq));
            fflush(stdout);
        }

        // Write the slab without the halo
        const auto result_start =                  0 == slab_start ? slab_start : slab_start + halo;
        const auto result_end   = mapping.n_images() == slab_end   ? slab_end   : slab_end   - halo;
        auto upd_cfg = cuda::make_config(
            cuda::grid_dims(dim3{
                ((unsigned int)(result_end - result_start) + (threads_per_block - 1)) / threads_per_block
              , mapping.extents().extent(1)
              , mapping.col_end() - mapping.col_begin()
              })
          , cuda::block_dims<threads_per_block>());
        cuda::launch(
            ctx.stream(), upd_cfg, update_kernel{}, mapping, d_slab_map
          , res->data(), V_ptrs, d_ci, slab_start, result_start, result_end);
    }
    fmt::print("\n");

    return res;
}

template <std::size_t... Ks>
constexpr auto instantiate_potts(std::index_sequence<Ks...>)
{
    return std::tuple{&potts_min_partition<Ks + 2>...};
}

[[maybe_unused]] constinit auto potts_instances =
    instantiate_potts(std::make_index_sequence<max_phases - 1>{});
