#include <cuda/algorithm>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/launch>
#include <cuda/utility>

#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

#include <layout_cylinder.h>

#include "grad_magn.h"

static constexpr auto threads_per_block = 256;

struct grad_kernel {
    template <typename Config, typename Extents>
    __device__
    void operator()(
        Config config
      , layout_cylinder::mapping<Extents> m
      , const float *u
      , float *v
      )
    {
        auto b   = cuda::block.index(cuda::grid, config);
        auto col = m.col_begin() + b.z;
        auto row = b.y;
        if (m.row_begin(col) <= row && row < m.row_end(col)) {
            auto n = cuda::gpu_thread.index(cuda::grid, config).x;

            if (m.n_images() > n) {
                if (
                    !(m.n_images() - 1 > n) || !(m.row_end(col) - 1 > row) ||
                    !(m.col_end() - 1 > col && m.row_begin(col + 1) <= row && row < m.row_end(col + 1)) ||
                    !(0 < n) ||
                    !(m.row_begin(col) < row) ||
                    !(m.col_begin() < col && m.row_begin(col - 1) <= row && row < m.row_end(col - 1))
                )
                {
                    v[m(n, row, col)] = -1.0f;
                } else {
                    auto q1 = 0.0f, q2 = 0.0f, q3 = 0.0f;

                    q1 += u[m(n + 1, row, col)];
                    q2 += u[m(n, row + 1, col)];
                    q3 += u[m(n, row, col + 1)];
                    q1 -= u[m(n - 1, row, col)];
                    q2 -= u[m(n, row - 1, col)];
                    q3 -= u[m(n, row, col - 1)];

                    v[m(n, row, col)] = 0.5f * cuda::std::sqrt(sqr(q1) + sqr(q2) + sqr(q3));
                }
            }
        }
    }
};

device_field_ptr<float>
grad_magn(CudaContext &ctx, DeviceField<float> &u) {
    auto mapping = u.mapping();

    auto v = device_field_ptr<float>{
        new DeviceField<float>{mapping, ctx.stream(), ctx.mr()}
      };

    auto config = cuda::make_config(
        cuda::grid_dims(dim3{
            (mapping.nstride() + (threads_per_block - 1)) / threads_per_block
          , mapping.extents().extent(1)
          , mapping.col_end() - mapping.col_begin()
          })
      , cuda::block_dims<threads_per_block>());

    cuda::launch(ctx.stream(), config, grad_kernel{}, mapping, u.data(), v->data());

    return v;
}
