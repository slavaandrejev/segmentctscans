#pragma once

#include <cuda-context.h>

device_field_ptr<float>
grad_magn(CudaContext &ctx, DeviceField<float> &u);
