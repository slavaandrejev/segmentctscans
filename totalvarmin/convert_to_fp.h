#pragma once

#include <cstdint>

#include <cuda-context.h>

device_field_ptr<float>
convert_to_fp(CudaContext &, DeviceField<uint16_t> &, uint16_t lo, uint16_t hi);
