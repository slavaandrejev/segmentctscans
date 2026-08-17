#pragma once

#include <cmath>
#include <concepts>
#include <cstdint>
#include <tuple>
#include <vector>

#include <fmt/printf.h>

#include <cuda-context.h>

void
chambolle(CudaContext &, DeviceField<float> &g, float 𝜆, float 𝜏, int iters);
