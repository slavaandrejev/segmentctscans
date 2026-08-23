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
#include <minsizetype.h>

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

    void lo(uint16_t x) { lo_ = x; }
    void hi(uint16_t x) { hi_ = x; }

    auto lo() const { return lo_; }
    auto hi() const { return hi_; }

private:
    static constexpr auto byte_alignment = 128;
    using allocator = boost::alignment::aligned_allocator<value_type, byte_alignment>;

    std::vector<value_type, allocator> storage;
    layout_cylinder::mapping_storage<extents_type> mapping_storage;

    uint16_t lo_{}; // if the storage is float, this is the original image level corresponding to 0.0f
    uint16_t hi_{}; // if the storage is float, this is the original image level corresponding to 1.0f

    template <typename Archive>
    void save(Archive &ar, const unsigned int) const {
        using namespace io;

        ar << mapping_storage;
        if constexpr (std::is_same_v<value_type, float>) {
            ar << u<16>(lo_);
            ar << u<16>(hi_);
        }
        auto span = view();
        auto m    = span.mapping();
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                    auto static constexpr nbits = sizeof(value_type) * 8;
                    using cont_type = decltype(MinSizeUInt<nbits>())::type;
                    auto cont = cont_type{};
                    std::memcpy(&cont, &span[n, row, col], nbits / 8);

                    ar << u<nbits>(cont);
                }
            }
        }
    }

    template <typename Archive>
    void load(Archive &ar, const unsigned int) {
        using namespace io;

        ar >> mapping_storage;
        storage.resize(mapping_storage.required_span_size());
        if constexpr (std::is_same_v<value_type, float>) {
            auto v = u<16>{};
            ar >> v;
            lo_ = uint16_t(v);
            ar >> v;
            hi_ = uint16_t(v);
        }
        auto span = view();
        auto m    = span.mapping();
        for (auto col = m.col_begin(); m.col_end() > col; ++col) {
            for (auto row = m.row_begin(col); m.row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; m.n_images() > n; ++n) {
                    auto static constexpr nbits = sizeof(value_type) * 8;
                    using cont_type = decltype(MinSizeUInt<nbits>())::type;
                    auto cont = cont_type{};

                    auto v = u<nbits>{};
                    ar >> v;
                    cont = cont_type(v);
                    std::memcpy(&span[n, row, col], &cont, nbits / 8);
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
