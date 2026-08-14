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

#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif

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
        auto mapping = layout_cylinder::mapping{extents_type{num_images, height, width}, xc, yc, r, stride};

        storage.resize(mapping.required_span_size());
        view = std::mdspan{storage.data(), std::move(mapping)};
    }

    explicit Field(const layout_cylinder::mapping<extents_type> &m)
      : storage(m.required_span_size())
      , view{storage.data(), m}
    {}

    Field(const Field &other)
      : storage(other.storage)
      , view{storage.data(), other.view.mapping()}
    {}

    Field & operator=(const Field &other) {
        storage = other.storage;
        view    = mdspan_type{storage.data(), other.view.mapping()};
        return *this;
    }

    Field(Field &&) noexcept = default;
    Field & operator=(Field &&) noexcept = default;

    auto const & mapping() const { return view.mapping(); }

    constexpr reference operator[](index_type n, index_type row, index_type col) const {
        return view[n, row, col];
    }

    constexpr reference operator[](index_type n, index_type row, index_type col) {
        return view[n, row, col];
    }

    auto row_begin(index_type col) const { return view.mapping().row_begin(col); }
    auto row_end(index_type col) const { return view.mapping().row_end(col); }

    auto col_begin() const { return view.mapping().col_begin(); }
    auto col_end()   const { return view.mapping().col_end(); }

    auto n_images() const { return view.mapping().n_images(); }

    auto nstride() const { return view.mapping().nstride(); }

     void swap(Field &other) noexcept {
        storage.swap(other.storage);
        std::swap(view, other.view);
    }

private:
    static constexpr auto byte_alignment = 128;
    using allocator = boost::alignment::aligned_allocator<value_type, byte_alignment>;

    std::vector<value_type, allocator> storage;
    mdspan_type view;

    template <typename Archive>
    void save(Archive &ar, const unsigned int) const {
        static_assert(std::is_integral_v<value_type>);
        using namespace io;

        ar << view.mapping();
        for (auto col = col_begin(); col_end() > col; ++col) {
            for (auto row = row_begin(col); row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; n_images() > n; ++n) {
                    using unsigned_value_type = std::make_unsigned_t<value_type>;
                    auto static constexpr nbits = sizeof(value_type) * 8;
                    ar << u<nbits>(unsigned_value_type((*this)[n, row, col]));
                }
            }
        }
    }

    template <typename Archive>
    void load(Archive &ar, const unsigned int) {
        static_assert(std::is_integral_v<value_type>);
        using namespace io;

        auto mapping = layout_cylinder::mapping<extents_type>{};
        ar >> mapping;
        storage.resize(mapping.required_span_size());
        view = mdspan_type{storage.data(), mapping};
        for (auto col = col_begin(); col_end() > col; ++col) {
            for (auto row = row_begin(col); row_end(col) > row; ++row) {
                for (auto n = uint32_t{}; n_images() > n; ++n) {
                    auto static constexpr nbits = sizeof(value_type) * 8;
                    auto v = u<nbits>{};
                    ar >> v;
                    (*this)[n, row, col] = value_type(v);
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
