#pragma once

#include <array>

#include <cuda-context.h>

static constexpr auto max_potts_labels = 6;

template <size_t K>
void potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , std::array<float, K> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  );
