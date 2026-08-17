#pragma once

#include <memory>

#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/stream>

#include <mdspan/mdspan.hpp>

#include <cuda-context.h>
#include <cuda-details/device-field-impl.h>

using mu16_span = std::mdspan<
    uint16_t
  , std::dextents<uint32_t, 3>
  , layout_cylinder
  >;
using mf32_span = std::mdspan<
    float
  , std::dextents<uint32_t, 3>
  , layout_cylinder
  >;

struct layout_cylinder;
class CudaContext {
public:
    CudaContext()
      : device_{cuda::devices[0]}
      , stream_{device_}
      , mr_{cuda::device_default_memory_pool(device_)}
    {}

    template <typename ElementType, typename Extents>
    device_field_ptr<ElementType>
    upload(std::mdspan<ElementType, Extents, layout_cylinder> a) {
        return device_field_ptr<std::remove_cv_t<ElementType>>{
            new DeviceField<std::remove_cv_t<ElementType>>{a, stream_, mr_}
        };
    }

    template <typename ElementType, typename Extents>
    void download(const DeviceField<ElementType> &,
                  std::mdspan<ElementType, Extents, layout_cylinder> out);

    auto stream()       { return cuda::stream_ref{stream_}; }
    auto stream() const { return cuda::stream_ref{stream_}; }

    auto mr()       { return mr_; }
    auto mr() const { return mr_; }

private:
    cuda::device_ref device_;
    cuda::stream     stream_;
    cuda::device_memory_pool_ref mr_;
};
