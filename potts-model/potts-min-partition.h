#pragma once

#include <span>

#include <cuda-context.h>

void potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , std::span<float> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  );
