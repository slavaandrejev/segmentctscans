#include <cstdint>

#include <cuda/launch>

#include <cuda-context.h>
#include <convert_to_fp.h>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

struct convert_to_fp_kernel {
    template <typename Config, typename ElementType>
    __device__
    void operator()(
        Config config
      , const std::mdspan<const ElementType
          , std::dextents<uint32_t, 3>
          , layout_cylinder
          > in
      , mf32_span out
      , float lo
      , float hi
      )
    {
        auto m = in.mapping();

        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            auto v = in[n, row, col];
            if (v <= lo) {
                out[n, row, col] = 0.0f;
                return;
            }
            if (v >= hi) {
                out[n, row, col] = 1.0f;
                return;
            }
            out[n, row, col] = float(v - lo) / (hi - lo);
        }
    }
};

template <typename ElementType>
device_field_ptr<float>
convert_to_fp(CudaContext &ctx, const DeviceField<ElementType> &in, float lo, float hi)
{
    auto mapping = in.mapping();

    auto out = device_field_ptr<float>{
        new DeviceField<float>{mapping, ctx.stream(), ctx.mr()}
      };

    constexpr int threads_per_block = 256;
    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            (mapping.nstride() + (threads_per_block - 1)) / threads_per_block
          , mapping.extents().extent(1)
          , mapping.col_end() - mapping.col_begin()
          })
      , cuda::block_dims<threads_per_block>());

    cuda::launch(ctx.stream(), config, convert_to_fp_kernel{}, in.view(), out->view(), lo, hi);

    return out;
}

template
device_field_ptr<float>
convert_to_fp(CudaContext &, const DeviceField<uint16_t> &, float lo, float hi);

template
device_field_ptr<float>
convert_to_fp(CudaContext &, const DeviceField<float> &, float lo, float hi);
