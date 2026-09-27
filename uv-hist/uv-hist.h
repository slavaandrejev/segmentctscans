#pragma once

#include <cstdint>
#include <tuple>
#include <vector>

#include <mdspan/mdspan.hpp>

#include <cuda-context.h>

constexpr auto hist_mapping_type() {
    return std::layout_stride::mapping(
        std::extents{uint16_t{}, uint16_t{}}
      , std::array<uint32_t, 2>{0, 0}
      );
}

struct UVHistReturn {
    std::vector<uint32_t> hist;
    decltype(hist_mapping_type()) mapping;
    float u_start;
    float u_end;
    float v_start;
    float v_end;
};

UVHistReturn
uv_hist(
    CudaContext &ctx
  , DeviceField<float> &u
  , DeviceField<float> &v
  , uint16_t nbins
  );

std::tuple<std::vector<uint32_t>, decltype(hist_mapping_type())>
uv_hist(
    CudaContext &ctx
  , DeviceField<float> &u
  , DeviceField<float> &v
  , float u_start
  , float u_end
  , float v_start
  , float v_end
  , uint16_t nbins);
