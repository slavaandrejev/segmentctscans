#pragma once

#include <cstdint>
#include <concepts>

#include <boost/mpl/int.hpp>
#include <boost/mpl/integral_c_tag.hpp>
#include <boost/serialization/level.hpp>

#include "minsizetype.h"

namespace io
{
    class b
    {
    public:
        b() = default;
        constexpr
        b(uint8_t v) : v(v) {}

        operator uint8_t() const { return v; }

    private:
        uint8_t v;
    };

    template <unsigned N, unsigned Value>
    class f
    {};

    template <unsigned N>
    class u
    {
    public:
        using type = decltype(MinSizeUInt<N>())::type;

        u() = default;
        constexpr
        u(type v) : v(v) {}

        operator type() const { return v; }

    private:
        type v;
    };

    template <>
    class u<1>
    {
    public:
        typedef bool type;

        u() = default;
        constexpr
        u(bool v) : v(v) {}

        explicit operator bool() const { return v; }

    private:
        bool v;
    };

    class uv
    {
    public:
        typedef uint64_t type;

        uv() = default;
        uv(const uv&) = default;
        uv(uv&&) = default;
        uv& operator=(const uv &) = default;
        uv& operator=(uv &&) = default;

        constexpr
        uv(uint64_t v) : v(v) {}

        operator uint64_t() const { return v; }

        template <std::integral T>
        uv& operator=(T t) {
            v = t;
            return *this;
        }

        uv& operator()(unsigned n) {
            nBits = n;
            return *this;
        }

        unsigned Bits() const {
            return nBits;
        }

    private:
        uint64_t v;
        unsigned nBits;
    };

    class ue
    {
    public:
        typedef uint64_t type;

        ue() = default;
        constexpr
        ue(type v) : v(v) {}

        operator type() const { return v; }

        template <typename IntegralType>
        ue& operator >>=(IntegralType rhs) {
          v >>= rhs;

          return *this;
        }

        ue& operator++() {
            ++v;
            return *this;
        }

        ue& operator--() {
            --v;
            return *this;
        }

        type operator++(int) {
            type save = v;
            ++v;
            return save;
        }

        type operator--(int) {
            type save = v;
            --v;
            return save;
        }
    private:
        type v;
    };

    class se
    {
    public:
        typedef int64_t type;

        se() = default;
        constexpr
        se(type v) : v(v) {}

        operator type() const { return v; }

    private:
        type v;
    };
}

namespace boost {
    namespace serialization {
        template <unsigned N, unsigned Value>
        struct implementation_level<io::f<N, Value>> {
            typedef mpl::integral_c_tag tag;
            typedef mpl::int_<primitive_type> type;
            static const int value = implementation_level::type::value;
        };

        template <unsigned N>
        struct implementation_level<io::u<N>> {
            typedef mpl::integral_c_tag tag;
            typedef mpl::int_<primitive_type> type;
            static const int value = implementation_level::type::value;
        };
    }
}

BOOST_CLASS_IMPLEMENTATION(io::b,  boost::serialization::primitive_type)
BOOST_CLASS_IMPLEMENTATION(io::uv, boost::serialization::primitive_type)
BOOST_CLASS_IMPLEMENTATION(io::ue, boost::serialization::primitive_type)
BOOST_CLASS_IMPLEMENTATION(io::se, boost::serialization::primitive_type)
