#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <cuda-context.h>

std::tuple<std::vector<double>, float, float>
calc_hist(CudaContext &ctx, DeviceField<float> &in, uint16_t nbins, float 𝛿);

std::vector<double>
calc_hist(CudaContext &ctx, DeviceField<float> &in, float range_start, float range_end, uint16_t nbins, float 𝛿);
