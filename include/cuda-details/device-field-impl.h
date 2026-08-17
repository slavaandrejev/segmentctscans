#pragma once

#include <concepts>
#include <type_traits>

#include <mdspan/mdspan.hpp>

#include <cuda/buffer>

#include <layout_cylinder.h>

template <typename ElementType>
    requires std::integral<std::remove_cv_t<ElementType>> || std::floating_point<std::remove_cv_t<ElementType>>
class DeviceField {
public:
    using element_type = ElementType;
    using extents_type = std::dextents<uint32_t, 3>;
    using mdspan_type  = std::mdspan<const element_type, extents_type, layout_cylinder>;
    using value_type   = mdspan_type::value_type;
    using layout_type  = mdspan_type::layout_type;
    using index_type   = mdspan_type::index_type;
    using mapping_type = layout_type::template mapping<extents_type>;

    DeviceField(mdspan_type a, cuda::stream_ref stream, cuda::device_memory_pool_ref mr)
      : d_voxels{
            stream
          , mr
          , a.data_handle()
          , a.data_handle() + a.mapping().required_span_size()
          }
      , d_mapping_tables{a.mapping().device_tables(stream, mr)}
      , d_mapping{a.mapping().with_tables(d_mapping_tables.csc())}
    {}

    DeviceField(const mapping_type &m, cuda::stream_ref stream, cuda::device_memory_pool_ref mr)
      : d_voxels{stream, mr, m.required_span_size(), cuda::no_init}
      , d_mapping_tables{m.device_tables(stream, mr)}
      , d_mapping{m.with_tables(d_mapping_tables.csc())}
    {}

    auto view()       { return std::mdspan{d_voxels.data(), d_mapping}; }
    auto view() const { return std::mdspan{d_voxels.data(), d_mapping}; }

    auto mapping()       { return d_mapping; }
    auto mapping() const { return d_mapping; }

private:
    cuda::device_buffer<value_type> d_voxels;
    mapping_type::template device_mapping_tables<index_type> d_mapping_tables;
    mapping_type d_mapping;
};
