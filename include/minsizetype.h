#pragma once

#include <boost/hana/type.hpp>

template <int neededBits>
consteval auto MinSizeUInt() {
    static_assert(neededBits <= 128, "Very long integer is requested");
    namespace hana = boost::hana;
    if constexpr (8 >= neededBits) {
        return hana::type_c<uint8_t>;
    } else if constexpr (16 >= neededBits) {
        return hana::type_c<uint16_t>;
    } else if constexpr (32 >= neededBits) {
        return hana::type_c<uint32_t>;
    } else if constexpr (64 >= neededBits) {
        return hana::type_c<uint64_t>;
    } else {
        return hana::type_c<unsigned __int128>;
    }
}

template <int neededBits>
consteval auto MinSizeInt() {
    static_assert(neededBits <= 128, "Very long integer is requested");
    namespace hana = boost::hana;
    if constexpr (8 >= neededBits) {
        return hana::type_c<int8_t>;
    } else if constexpr (16 >= neededBits) {
        return hana::type_c<int16_t>;
    } else if constexpr (32 >= neededBits) {
        return hana::type_c<int32_t>;
    } else if constexpr (64 >= neededBits) {
        return hana::type_c<int64_t>;
    } else {
        return hana::type_c<__int128>;
    }
}
