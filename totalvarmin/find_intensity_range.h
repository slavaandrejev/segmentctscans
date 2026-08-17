#pragma once

#include <concepts>
#include <cstdint>
#include <tuple>

#include <cuda-context.h>

std::tuple<uint16_t, uint16_t>
find_intensity_range(CudaContext &, DeviceField<uint16_t> &, double tail_thr);
