#pragma once

#include <concepts>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <boost/align/aligned_allocator.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/wrapper.hpp>

#include <mdspan/mdspan.hpp>

#include <io/primitivetypes.h>
#include <layout_cylinder.h>

template <typename ElementType>
    requires std::integral<std::remove_cv_t<ElementType>> || std::floating_point<std::remove_cv_t<ElementType>>
class Field {
    friend class boost::serialization::access;
    friend void swap(Field &a, Field &b) noexcept { a.swap(b); }
public:
    using element_type = ElementType;
    using extents_type = std::dextents<uint32_t, 3>;
    using mdspan_type  = std::mdspan<element_type, extents_type, layout_cylinder>;
    using value_type   = mdspan_type::value_type;
    using index_type   = mdspan_type::index_type;
    using reference    = mdspan_type::reference;

    Field() = default;

    Field(uint32_t num_images, uint32_t width, uint32_t height, double xc, double yc, double r) {
        static auto constexpr align = uint32_t(byte_alignment / sizeof(value_type));
        const auto stride = uint32_t(((num_images + (align - 1)) / align) * align);
        mapping_storage = layout_cylinder::mapping_storage{
            extents_type{num_images, height, width}
          , xc
          , yc
          , r
          , stride
          };

        storage.resize(mapping_storage.required_span_size());
    }

    explicit Field(const layout_cylinder::mapping<extents_type> &m)
      : storage(m.required_span_size())
      , mapping_storage(m)
    {}

    auto view() { return std::mdspan{storage.data(), mapping_storage.mapping()}; }
    auto view() const { return std::mdspan{storage.data(), mapping_storage.mapping()}; }

    void swap(Field &other) noexcept {
        storage.swap(other.storage);
        std::swap(mapping_storage, other.mapping_storage);
    }

private:
    static constexpr auto byte_alignment = 128;
    using allocator = boost::alignment::aligned_allocator<value_type, byte_alignment>;

    std::vector<value_type, allocator> storage;
    layout_cylinder::mapping_storage<extents_type> mapping_storage;

    template <typename Archive>
    void save(Archive &ar, const unsigned int) const {
        static_assert(std::is_integral_v<value_type>);
        using namespace io;

        ar << mapping_storage;
        auto span = view();
        auto m    = span.mapping();
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                    using unsigned_value_type = std::make_unsigned_t<value_type>;
                    auto static constexpr nbits = sizeof(value_type) * 8;
                    ar << u<nbits>(unsigned_value_type(span[n, row, col]));
                }
            }
        }
    }

    template <typename Archive>
    void load(Archive &ar, const unsigned int) {
        static_assert(std::is_integral_v<value_type>);
        using namespace io;

        ar >> mapping_storage;
        storage.resize(mapping_storage.required_span_size());
        auto span = view();
        auto m    = span.mapping();
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                    auto static constexpr nbits = sizeof(value_type) * 8;
                    auto v = u<nbits>{};
                    ar >> v;
                    span[n, row, col] = value_type(v);
                }
            }
        }
    }

    BOOST_SERIALIZATION_SPLIT_MEMBER()
};

namespace boost {
    namespace serialization {
        template <typename ElementType>
        struct implementation_level<Field<ElementType>> {
            typedef mpl::integral_c_tag tag;
            typedef mpl::int_<object_serializable> type;
            BOOST_STATIC_CONSTANT(int, value = implementation_level::type::value);
        };

        template <typename ElementType>
        struct tracking_level<Field<ElementType>> {
            typedef mpl::integral_c_tag tag;
            typedef mpl::int_<track_never> type;
            BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
        };
    }
}
