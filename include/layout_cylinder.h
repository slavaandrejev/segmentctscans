#pragma once

#if defined(__cpp_lib_mdspan)
#include <mdspan>
#else
#include <mdspan/mdspan.hpp>
#endif

#include <cmath>
#include <vector>

#include <boost/serialization/access.hpp>
#include <boost/serialization/level.hpp>
#include <boost/serialization/split_member.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/wrapper.hpp>

#include <io/primitivetypes.h>

#include <sqr.h>

struct layout_cylinder {
    template <typename Index>
    struct mapping_tables {
        const Index *start_row_{};
        const Index *base_{};
    };

    template <class Extents>
    struct mapping {
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using size_type    = typename Extents::size_type;
        using rank_type    = typename Extents::rank_type;
        using layout_type  = layout_cylinder;

        Extents ext_{};

        index_type nstride_{};
        index_type col_begin_{};
        index_type col_end_{};
        index_type span_size_{};

        mapping_tables<index_type> csc{};

        constexpr auto operator()(index_type n, index_type row, index_type col) const -> index_type
        {
            return csc.base_[col] + (row - csc.start_row_[col]) * nstride_ + n;
        }

        constexpr auto required_span_size() const -> index_type { return span_size_; }

        constexpr auto extents() const -> const Extents & { return ext_; }

        constexpr auto row_begin(index_type col) const { return csc.start_row_[col]; }
        constexpr auto row_end(index_type col) const {
            return csc.start_row_[col] + (csc.base_[col + 1] - csc.base_[col]) / nstride_;
        }

        constexpr auto col_begin() const { return col_begin_; }
        constexpr auto col_end() const { return col_end_; }

        constexpr auto n_images() const { return ext_.extent(0); }

        constexpr auto nstride() const { return nstride_; }

        static constexpr bool is_always_unique()     { return true; }
        static constexpr bool is_always_exhaustive() { return false; }
        static constexpr bool is_always_strided()    { return false; }
        static constexpr bool is_unique()            { return true; }
               constexpr bool is_exhaustive() const  { return ext_.extent(0) == nstride_; }
        static constexpr bool is_strided()           { return false; }

        friend constexpr bool
        operator==(const mapping &, const mapping &) = default;
    };

    // This class owns the mapping data
    template <class Extents>
    class mapping_storage {
        friend class boost::serialization::access;
    public:
        using extents_type = Extents;
        using index_type   = typename Extents::index_type;
        using size_type    = typename Extents::size_type;
        using layout_type  = layout_cylinder;

        mapping_storage() = default;

        mapping_storage(const layout_type::mapping<extents_type> &other)
          : ext_{other.ext_}
          , nstride_{other.nstride_}
          , col_begin_{other.col_begin_}
          , col_end_{other.col_end_}
          , start_row_(other.csc.start_row_, other.csc.start_row_ + ext_.extent(2))
          , base_(other.csc.base_, other.csc.base_ + ext_.extent(2) + 1)
        {}

        // Extents are: number of images, rows and columns in the original TIFF
        mapping_storage(const Extents &e, double xc, double yc, double r, index_type nstride)
          : ext_{e}
          , nstride_{nstride}
          , start_row_(e.extent(2))
          , base_(e.extent(2) + 1)
        {
            auto const r2 = r * r;
            auto off = index_type{};
            for (auto col = index_type{}; e.extent(2) > col; ++col) {
                auto 𝛿2 = r2 - sqr(double(col) - xc);
                base_[col] = off;
                auto r0 = 0.0, r1 = 0.0;
                if (0 < 𝛿2) {
                    auto 𝛿 = std::sqrt(𝛿2);
                    auto c = std::ceil(yc - 𝛿);
                    if (yc - 𝛿 == c) c += 1.0;
                    auto f = std::floor(yc + 𝛿);
                    if (yc + 𝛿 == f) f -= 1.0;
                    r0 = std::max(0.0, c);
                    r1 = std::min(double(e.extent(1) - 1), f);
                }
                if (0 < 𝛿2 && r0 <= r1) {
                    auto rr0 = index_type(r0);
                    auto rr1 = index_type(r1);

                    if (~index_type{} == col_begin_) {
                        col_begin_ = col;
                        col_end_   = col + 1;
                    } else {
                        ++col_end_;
                    }

                    start_row_[col] = rr0;
                    off            += (rr1 - rr0 + 1) * nstride_;
                } else {
                    start_row_[col] = ~index_type{};
                }
            }
            base_[e.extent(2)] = off; // total size
        }

        auto mapping(mapping_tables<index_type> csc) const {
            return layout_type::mapping<extents_type>{
                .ext_       = ext_
              , .nstride_   = nstride_
              , .col_begin_ = col_begin_
              , .col_end_   = col_end_
              , .span_size_ = required_span_size()
              , .csc        = csc
              };
        }

        auto mapping() const {
            return mapping(mapping_tables<index_type>{
                .start_row_ = start_row_.data()
              , .base_      = base_.data()
              });
        }

        auto required_span_size() const -> index_type { return base_.back(); }

private:
        extents_type ext_{};
        index_type   nstride_{};
        index_type   col_begin_{~index_type{}};
        index_type   col_end_{};
        std::vector<index_type> start_row_{};
        std::vector<index_type> base_{}; // per-col base offset; last = total

        template <typename Archive>
        void save(Archive &ar, const unsigned int) const {
            using namespace io;

            ar << ue(ext_.extent(0));
            ar << ue(ext_.extent(1));
            ar << ue(ext_.extent(2));

            ar << ue(nstride_);
            ar << ue(col_begin_);
            ar << ue(col_end_);

            ar << ue(start_row_.size());
            if (!start_row_.empty()) {
                ar << ue(start_row_[0]);
                for (auto i = size_t{1}; start_row_.size() > i; ++i) {
                    ar << se(int64_t(start_row_[i]) - int64_t(start_row_[i - 1]));
                }
            }
            ar << ue(base_.size());
            if (!base_.empty()) {
                ar << ue(base_[0]);
                for (auto i = size_t{1}; base_.size() > i; ++i) {
                    ar << se(int64_t(base_[i]) - int64_t(base_[i - 1]));
                }
            }
        }

        template <typename Archive>
        void load(Archive &ar, const unsigned int) {
            using namespace io;

            auto e1 = ue{};
            auto e2 = ue{};
            auto e3 = ue{};

            ar >> e1;
            ar >> e2;
            ar >> e3;

            ext_ = extents_type{index_type(e1), index_type(e2), index_type(e3)};

            auto nstride   = ue{};
            auto col_begin = ue{};
            auto col_end   = ue{};

            ar >> nstride;
            ar >> col_begin;
            ar >> col_end;

            nstride_   = index_type(nstride);
            col_begin_ = index_type(col_begin);
            col_end_   = index_type(col_end);

            auto start_row_size = ue{};
            ar >> start_row_size;
            if (0 < start_row_size) {
                start_row_.resize(start_row_size);
                auto v = ue{};
                ar >> v;
                start_row_[0] = v;
                for (auto i = size_t{1}; start_row_.size() > i; ++i) {
                    auto s = se{};
                    ar >> s;
                    start_row_[i] = index_type(start_row_[i - 1] + s);
                }
            }
            auto base_size = ue{};
            ar >> base_size;
            if (0 < base_size) {
                base_.resize(base_size);
                auto v = ue{};
                ar >> v;
                base_[0] = v;
                for (auto i = size_t{1}; base_.size() > i; ++i) {
                    auto s = se{};
                    ar >> s;
                    base_[i] = index_type(base_[i - 1] + s);
                }
            }
        }

        BOOST_SERIALIZATION_SPLIT_MEMBER()
    };
};

namespace boost {
    namespace serialization {
        template <typename Extents>
        struct implementation_level<layout_cylinder::mapping_storage<Extents>> {
            typedef mpl::integral_c_tag tag;
            typedef mpl::int_<object_serializable> type;
            BOOST_STATIC_CONSTANT(int, value = implementation_level::type::value);
        };

        template <typename Extents>
        struct tracking_level<layout_cylinder::mapping_storage<Extents>> {
            typedef mpl::integral_c_tag tag;
            typedef mpl::int_<track_never> type;
            BOOST_STATIC_CONSTANT(int, value = tracking_level::type::value);
        };
    }
}
