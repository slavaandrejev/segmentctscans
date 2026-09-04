#pragma once

#include <tuple>

#include <cuda-context.h>

std::tuple<float, float> min_max(CudaContext &ctx, const DeviceField<float> &in);
