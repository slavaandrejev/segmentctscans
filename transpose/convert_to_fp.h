#pragma once

#include <concepts>
#include <cstdint>

#include <cuda-context.h>

template <typename ElementType>
device_field_ptr<float>
convert_to_fp(CudaContext &, const DeviceField<ElementType> &, float lo, float hi);
