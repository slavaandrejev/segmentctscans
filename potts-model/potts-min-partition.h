#pragma once

#include <array>

#include <cuda-context.h>
#include <field.h>

template <size_t K>
device_field_ptr<float> potts_min_partition(
    CudaContext &ctx
  , DeviceField<float> &g
  , const Field<float>::mapping_type &cpu_map
  , std::array<float, K> ci
  , float 𝜆, float 𝜏1, float 𝜏2
  , int iters
  );
