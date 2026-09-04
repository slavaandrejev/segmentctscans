#include <array>
#include <limits>
#include <tuple>

#include <cub/block/block_reduce.cuh>
#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/launch>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include "min_max.h"

constexpr auto threads_per_block = 256;

struct min_max_kernel {
    template <typename Config>
    __device__
    void operator()(
        Config config
      , const cmf32_span in
      , cuda::std::span<float> minmax
      )
    {
        using BlockReduce = cub::BlockReduce<float, threads_per_block>;
        __shared__ typename BlockReduce::TempStorage max_temp;
        __shared__ typename BlockReduce::TempStorage min_temp;

        auto m = in.mapping();

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        auto max = std::numeric_limits<float>::lowest();
        auto min = std::numeric_limits<float>::max();
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                max = min = in[n, row, col];
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

std::tuple<float, float> min_max(CudaContext &ctx, const DeviceField<float> &in) {
    auto mapping = in.mapping();

    auto res = std::array{
        std::numeric_limits<float>::max()
      , std::numeric_limits<float>::lowest()
      };
    auto d_res = cuda::device_buffer<float>{ctx.stream(), ctx.mr(), res};

    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            (mapping.nstride() + (threads_per_block - 1)) / threads_per_block
          , mapping.extents().extent(1)
          , mapping.col_end() - mapping.col_begin()
          })
      , cuda::block_dims<threads_per_block>());

    cuda::launch(ctx.stream(), config, min_max_kernel{}, in.view(), d_res);

    cuda::copy_bytes(ctx.stream(), d_res, cuda::std::span(&res[0], 2));
    ctx.stream().sync();

    return {res[0], res[1]};
}
