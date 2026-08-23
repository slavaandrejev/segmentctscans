#pragma once

#include <cstdint>
#include <vector>

#include <cuda-context.h>

std::vector<double>
calc_hist(CudaContext &ctx, DeviceField<float> &in, float 𝛿, uint16_t nbins);
