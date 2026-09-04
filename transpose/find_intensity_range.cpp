#include <cstdint>
#include <tuple>

#include <mdspan/mdspan.hpp>

#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>

#include <range/v3/numeric/accumulate.hpp>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include "find_intensity_range.h"

static auto constexpr num_hists         = 16;
static auto constexpr hist_size         = 1 << (sizeof(uint16_t) * 8);
static auto constexpr threads_per_block = 256;

struct histogram_kernel {
    template <typename Config>
    __device__ void operator()(
        Config config
      , const mu16_span in
      , cuda::std::span<uint32_t> block_hists)
    {
        auto m = in.mapping();

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto hist_idx = cuda::block.rank(cuda::grid, config) % num_hists;
            auto hist     = block_hists.subspan(hist_idx * hist_size, hist_size);

            auto stride = cuda::gpu_thread.dims(cuda::grid, config).x;
            for (auto n = cuda::gpu_thread.index(cuda::grid, config).x; m.n_images() > n; n += stride)
            {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{hist[in[n, row, col]]}
                    .fetch_add(1, cuda::std::memory_order_relaxed);
            }
        }
    }
};

struct histogram_reduce_kernel {
    template <typename Config>
    __device__ void operator()(
        Config config
      , cuda::std::span<const uint32_t> block_hists
      , cuda::std::span<uint64_t> hist)
    {
        auto bin = cuda::gpu_thread.rank(cuda::grid, config);
        if (hist.size() > bin) {
            auto sum = uint64_t{};
            for (auto c = 0; num_hists > c; ++c) {
                sum += block_hists[c * hist_size + bin];
            }
            hist[bin] = sum;
        }
    }
};

std::tuple<uint16_t, uint16_t>
find_intensity_range(CudaContext &ctx, DeviceField<uint16_t> &field, double tail_thr)
{
    namespace rs = ranges;

    auto d_hists = cuda::device_buffer<uint32_t>{ctx.stream(), ctx.mr(), hist_size * num_hists, cuda::no_init};
    cuda::fill_bytes(ctx.stream(), d_hists, 0);

    auto view    = field.view();
    auto mapping = view.mapping();

    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            1
          , mapping.extents().extent(1)
          , mapping.col_end() - mapping.col_begin()
          })
      , cuda::block_dims<threads_per_block>());
    cuda::launch(ctx.stream(), config, histogram_kernel{}, view, d_hists);

    auto d_hist = cuda::device_buffer<uint64_t>{ctx.stream(), ctx.mr(), hist_size, cuda::no_init};

    auto reduce_config = cuda::make_config(
        cuda::grid_dims(dim3{(hist_size + (threads_per_block - 1)) / threads_per_block, 1, 1})
      , cuda::block_dims<threads_per_block>());
    cuda::launch(ctx.stream(), reduce_config, histogram_reduce_kernel{}, d_hists, d_hist);

    auto hist = std::vector<uint64_t>(hist_size);
    cuda::copy_bytes(ctx.stream(), d_hist, hist);
    ctx.stream().sync();

    auto total = rs::accumulate(hist, uint64_t{});

    auto const pct_lo = tail_thr * total;
    auto const pct_hi = (1.0 - tail_thr) * total;
    auto lo = size_t{}, hi = hist.size();
    auto cum = uint64_t{};
    for (auto i = size_t{}; hist.size() > i; ++i) {
        cum += hist[i];
        if (pct_lo >= cum) { lo = i; }
        if (pct_hi <= cum) { hi = i; break; }
    }

    return std::tuple{uint16_t(lo), uint16_t(hi)};
}
