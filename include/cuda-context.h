#pragma once

#include <memory>

#include <mdspan/mdspan.hpp>

struct layout_cylinder;

class CudaContext;
struct CudaContextDeleter { static void operator()(CudaContext *) noexcept; };
using cuda_context_ptr = std::unique_ptr<CudaContext, CudaContextDeleter>;
cuda_context_ptr make_context();

template <typename ElementType>
    requires std::integral<std::remove_cv_t<ElementType>> || std::floating_point<std::remove_cv_t<ElementType>>
class DeviceField;
template <typename ElementType>
struct DeviceFieldDeleter { static void operator()(DeviceField<ElementType> *) noexcept; };
template <typename ElementType>
using device_field_ptr = std::unique_ptr<DeviceField<ElementType>, DeviceFieldDeleter<ElementType>>;

template <typename ElementType, typename Extents>
device_field_ptr<ElementType>
upload(CudaContext &,
       std::mdspan<ElementType, Extents, layout_cylinder> a);

template <typename ElementType, typename Extents>
void download(CudaContext &, const DeviceField<ElementType> &,
              std::mdspan<ElementType, Extents, layout_cylinder> out);

struct device_memory_info {
    std::size_t available;
    std::size_t total;
};
device_memory_info device_memory(CudaContext const& ctx);
