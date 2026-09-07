#pragma once

#include <mdspan/mdspan.hpp>

struct layout_strict_upper {
    template <class Extents>
    struct mapping {
        using layout_type  = layout_strict_upper;
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using size_type    = typename Extents::size_type;
        using rank_type    = typename Extents::rank_type;

        extents_type ext_{};

        mapping() = default;

        mapping(const layout_strict_upper::mapping<extents_type> &other)
          : ext_{other.ext_}
        {}

        mapping(const extents_type &e)
          : ext_(e)
        {}

        constexpr auto operator()(index_type i, index_type j) const -> index_type {
            return (2 * ext_.extent(0) - 1 - i) * i / 2 + j - i - 1;
        }

        constexpr index_type required_span_size() const {
            auto n = ext_.extent(0);
            return n * (n - 1) / 2;
        }

        constexpr const extents_type& extents() const { return ext_; }

        static constexpr bool is_always_unique()     { return true; }
        static constexpr bool is_always_exhaustive() { return true; }
        static constexpr bool is_always_strided()    { return false; }
        static constexpr bool is_unique()            { return true; }
        static constexpr bool is_exhaustive()        { return true; }
        static constexpr bool is_strided()           { return false; }

        friend constexpr bool
        operator==(const mapping &, const mapping &) = default;
    };
};
template <typename T>
using strict_upper_span = std::mdspan<T, std::dextents<uint32_t, 2>, layout_strict_upper>;
