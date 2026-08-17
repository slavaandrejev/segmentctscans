#include <cstdint>
#include <random>
#include <vector>

#include <range/v3/numeric/accumulate.hpp>
#include <range/v3/algorithm/copy.hpp>
#include <range/v3/view/transform.hpp>

#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>
#include <cuda/std/cmath>
#include <cuda/std/random>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include <calc_hist.h>

struct calc_hist_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , const mf32_span in
      , cuda::std::span<uint32_t> global_hist
      , float 𝛿
      , uint16_t nbins
      , unsigned int seed
      )
    {
        auto s_hist = cuda::dynamic_shared_memory(config);

        for (auto i = cuda::gpu_thread.rank(cuda::block, config); nbins > i;
            i += cuda::gpu_thread.count(cuda::block, config))
        {
            s_hist[i] = 0;
        }
         __syncthreads();

        auto m = in.mapping();


        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto rng = cuda::std::philox4x32{seed};
            rng.set_counter({0, cuda::gpu_thread.rank(cuda::grid, config), 0, 0});
            auto dist = cuda::std::uniform_real_distribution{-𝛿 / 2, 𝛿 / 2};

            auto stride = cuda::gpu_thread.dims(cuda::grid, config).x;
            for (auto n = cuda::gpu_thread.index(cuda::grid, config).x; m.n_images() > n; n += stride)
            {
                auto v = in[n, row, col] + dist(rng);
                auto bin = int32_t(cuda::std::round(nbins * v - 0.5f));
                bin = cuda::std::max(0, cuda::std::min(int(nbins) - 1, bin));
                cuda::atomic_ref<uint32_t, cuda::thread_scope_block>{s_hist[bin]}
                    .fetch_add(1, cuda::std::memory_order_relaxed);
            }
        }
        __syncthreads();

        for (auto i = cuda::gpu_thread.rank(cuda::block, config); nbins > i;
            i += cuda::gpu_thread.count(cuda::block, config))
        {
            if (0 != s_hist[i]) {
                cuda::atomic_ref<uint32_t, cuda::thread_scope_device>{global_hist[i]}
                    .fetch_add(s_hist[i], cuda::std::memory_order_relaxed);
            }
        }
    }
};

std::vector<double> calc_hist(CudaContext &ctx, DeviceField<float> &in, float 𝛿, uint16_t nbins)
{
    namespace rs = ranges;
    namespace rv = rs::views;

    auto mapping = in.mapping();

    auto d_hist = cuda::device_buffer<uint32_t>{ctx.stream(), ctx.mr(), nbins, cuda::no_init};
    cuda::fill_bytes(ctx.stream(), d_hist, 0);

    constexpr int threads_per_block = 256;
    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            1
          , mapping.extents().extent(1)
          , mapping.col_end() - mapping.col_begin()
          })
      , cuda::block_dims<threads_per_block>()
      , cuda::dynamic_shared_memory<uint32_t[]>(nbins));

    cuda::launch(ctx.stream(), config, calc_hist_kernel{}, in.view(), d_hist, 𝛿, nbins, std::random_device{}());

    auto hist = std::vector<uint32_t>(nbins);
    cuda::copy_bytes(ctx.stream(), d_hist, hist);
    ctx.stream().sync();

    auto total = rs::accumulate(hist, uint64_t{}) / double(nbins);
    auto fp_hist = std::vector<double>(nbins, 0);
    rs::copy(rv::transform(hist, [&](auto v) { return v / total; }), fp_hist.begin());

    return fp_hist;
}
