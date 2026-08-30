#include <cstdint>

#include <mdspan/mdspan.hpp>

#include <cuda/algorithm>
#include <cuda/std/span>

#include <layout_cylinder.h>

#include <cuda-context.h>
#include <cuda-details/cuda-context-impl.h>
#include <cuda-details/device-field-impl.h>

void CudaContextDeleter::operator()(CudaContext *context) noexcept {
    delete context;
}

cuda_context_ptr make_context() {
    return cuda_context_ptr{new CudaContext{}};
}

template <typename ElementType>
void DeviceFieldDeleter<ElementType>::operator()(DeviceField<ElementType> *field) noexcept {
    delete field;
}

template <typename ElementType, typename Extents>
device_field_ptr<ElementType>
upload(
    CudaContext &context
  , std::mdspan<ElementType, Extents, layout_cylinder> a
  )
{
    return context.upload(a);
}

template <typename ElementType, typename Extents>
void download(
    CudaContext &context
  , const DeviceField<ElementType> &field
  , std::mdspan<ElementType, Extents, layout_cylinder> out
  )
{
    auto in = field.view();
    auto in_mapping = in.mapping();
    auto out_mapping = out.mapping();
    if (in_mapping.required_span_size() != out_mapping.required_span_size()) {
        throw std::runtime_error("Host and device images have different sizes.");
    }
    cuda::copy_bytes(
        context.stream()
      , cuda::std::span{in.data_handle(),  in.data_handle()  + in_mapping.required_span_size()}
      , cuda::std::span{out.data_handle(), out.data_handle() + out_mapping.required_span_size()});
    context.stream().sync();
}

template class DeviceField<uint16_t>;
template class DeviceField<float>;
template struct DeviceFieldDeleter<uint16_t>;
template struct DeviceFieldDeleter<float>;

template
device_field_ptr<uint16_t> upload(CudaContext &, mu16_span);
template
device_field_ptr<float> upload(CudaContext&, mf32_span);
template
void download(CudaContext &, const DeviceField<uint16_t> &, mu16_span out);
template
void download(CudaContext &, const DeviceField<float> &, mf32_span out);
