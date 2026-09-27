#include <cstdint>
#include <limits>

#include <cub/device/device_histogram.cuh>
#include <cub/block/block_reduce.cuh>
#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/launch>
#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cmath>

#include <mdspan/mdspan.hpp>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include <min_max.h>

#include "uv-hist.h"

constexpr auto byte_alignment = 128;
constexpr auto threads_per_block = 256;

using value_type = uint32_t;

// A special version of finding minimum and maximum for the gradient magnitude
// that searches only strictly positive values.
struct grad_min_max_kernel {
    template <typename Config, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , const float *v
      , cuda::std::span<float> minmax
      )
    {
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;
        __shared__ typename BlockReduce::TempStorage max_temp;
        __shared__ typename BlockReduce::TempStorage min_temp;

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto max = std::numeric_limits<float>::lowest();
        auto min = std::numeric_limits<float>::max();
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            const auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                const auto x = v[m(n, row, col)];
                if (0 < x) {
                    max = min = x;
                }
            }
        }
        auto block_min = BlockReduce(min_temp).Reduce(min, cuda::minimum<>{});
        if (0 == cuda::gpu_thread.rank(cuda::block, config)) {
            cuda::atomic_ref<float, cuda::thread_scope_device>{minmax[0]}
                .fetch_min(block_min, cuda::std::memory_order_relaxed);
        }
        auto block_max = BlockReduce(max_temp).Reduce(max, cuda::maximum<>{});
        if (0 == cuda::gpu_thread.rank(cuda::block, config)) {
            cuda::atomic_ref<float, cuda::thread_scope_device>{minmax[1]}
                .fetch_max(block_max, cuda::std::memory_order_relaxed);
        }
    }
};

UVHistReturn
uv_hist(CudaContext &ctx, DeviceField<float> &u, DeviceField<float> &v, uint16_t nbins)
{
    auto m = v.mapping();

    auto vminmax = std::array{
        std::numeric_limits<float>::max()
      , std::numeric_limits<float>::lowest()
      };
    auto d_vminmax = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), vminmax};
    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            (m.nstride() + (threads_per_block - 1)) / threads_per_block
          , m.extents().extent(1)
          , m.col_end() - m.col_begin()
          })
      , cuda::block_dims<threads_per_block>());
    cuda::launch(ctx.stream(), config, grad_min_max_kernel{}, m, v.data(), d_vminmax);
    cuda::copy_bytes(ctx.stream(), d_vminmax, cuda::std::span(&vminmax[0], 2));
    ctx.stream().sync();

    auto [u_min, u_max] = min_max(ctx, u);
    auto [v_min, v_max] = vminmax;
    auto [hist, mapping] = uv_hist(ctx, u, v, u_min, u_max, v_min, v_max, nbins);

    return {hist, mapping, u_min, u_max, v_min, v_max};
}

std::tuple<std::vector<uint32_t>, decltype(hist_mapping_type())>
uv_hist(
    CudaContext &ctx
  , DeviceField<float> &u
  , DeviceField<float> &v
  , float u_start
  , float u_end
  , float v_start
  , float v_end
  , uint16_t nbins
  )
{
    auto mapping = u.mapping();

    static auto constexpr align = byte_alignment / sizeof(value_type);
    const auto stride = value_type(((nbins + (align - 1)) / align) * align);

    auto hist_map = cuda::std::layout_stride::mapping(
        cuda::std::extents{nbins, nbins}
      , cuda::std::array<uint32_t, 2>{stride, 1}
      );

    auto cit = cuda::counting_iterator{0};
    auto uit = cuda::transform_iterator{
        cit
      , [data = u.data(), n = mapping.n_images(), stride = mapping.nstride()] __host__ __device__ (auto c) {
            return data[(c / n) * stride + c % n];
        }
      };
    auto vit = cuda::transform_iterator{
        cit
      , [data = v.data(), n = mapping.n_images(), stride = mapping.nstride()] __host__ __device__ (auto c) {
            return data[(c / n) * stride + c % n];
        }
      };
    auto bin_ids = cuda::zip_transform_iterator{
        [=] __host__ __device__(float x, float y) -> int {
            if (!(u_start <= x && x <= u_end &&
                v_start <= y && y <= v_end))
                return -1;

            auto ext = hist_map.extents();

            const auto tx = (x - u_start) / (u_end - u_start);
            const auto ty = (y - v_start) / (v_end - v_start);

            auto xbin = int32_t(cuda::std::round(ext.extent(1) * tx - 0.5f));
            auto ybin = int32_t(cuda::std::round(ext.extent(0) * ty - 0.5f));
            xbin = cuda::std::max(0, cuda::std::min(int(ext.extent(1)) - 1, xbin));
            ybin = cuda::std::max(0, cuda::std::min(int(ext.extent(0)) - 1, ybin));

            return int(hist_map(ybin, xbin));
        }
      , uit
      , vit
      };

    auto d_hist = cuda::device_buffer<value_type>{ctx.stream(), ctx.mr(), hist_map.required_span_size(), cuda::no_init};
    cuda::fill_bytes(ctx.stream(), d_hist, 0);

    auto temp_storage_bytes = size_t{};
    cub::DeviceHistogram::HistogramEven(
        nullptr
      , temp_storage_bytes
      , bin_ids
      , d_hist.data()
      , hist_map.required_span_size() + 1
      , value_type(0)
      , value_type(hist_map.required_span_size())
      , mapping.data_size()
      , ctx.stream().get()
      );

    auto temp_storage = cuda::device_buffer<uint8_t>{ctx.stream(), ctx.mr(), temp_storage_bytes, cuda::no_init};
    cub::DeviceHistogram::HistogramEven(
        temp_storage.data()
      , temp_storage_bytes
      , bin_ids
      , d_hist.data()
      , hist_map.required_span_size() + 1
      , value_type(0)
      , value_type(hist_map.required_span_size())
      , mapping.data_size()
      , ctx.stream().get()
      );

    auto hist = std::vector<uint32_t>(hist_map.required_span_size());
    cuda::copy_bytes(ctx.stream(), d_hist, hist);
    ctx.stream().sync();

    auto ext = hist_map.extents();
    auto std_hist_map = std::layout_stride::mapping(
        std::extents{uint16_t(ext.extent(0)), uint16_t(ext.extent(1))}
      , std::array<uint32_t, 2>{stride, 1}
      );

    return {hist, std_hist_map};
}
