#pragma once

#include <cuda-context.h>

void
chambolle(CudaContext &, DeviceField<float> &g, float 𝜆, float 𝜏, int iters);
