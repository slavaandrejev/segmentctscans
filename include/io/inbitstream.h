#pragma once

#include <concepts>
#include <cstdint>
#include <iterator>
#include <type_traits>
#include <utility>

#include "minsizetype.h"

namespace io
{
    template <typename InputIterator>
        requires (
            std::input_iterator<InputIterator>
         && std::integral<std::iter_value_t<InputIterator>>
         && 1 == sizeof(std::iter_value_t<InputIterator>))
    class InputBitStream
    {
    public:
        InputBitStream(InputIterator start, InputIterator finish)
          : it(start)
          , end(finish)
        {}

        operator bool() {
            return end != it;
        }

        auto IsByteAligned() const {
            return 0 == curBit;
        }

        template <unsigned MaxSize>
        auto Read(unsigned n = MaxSize) -> decltype(MinSizeUInt<MaxSize>())::type {
            if (end == it) {
                return 0;
            }
            using ResType = decltype(MinSizeUInt<MaxSize>())::type;
            const auto tBits    = curBit + n;          // total amount of bits to read
                  auto nBytes   = (tBits - 1) / 8 + 1; // the amount of bytes that hold the bits we are reading
            const auto bitsLeft = nBytes * 8 - tBits;  // remaining bits belonging to the next read inside `nBytes`
            ResType res = 0;
            ResType preRead = 0;
            // If we don't fit into the word, read bits to the end of the current
            // byte first.
            if (nBytes > sizeof(res)) {
                unsigned prereadBits = 8 - curBit;
                preRead = Read<8>(prereadBits);
                if (end == it) {
                    return preRead;
                }
                --nBytes;
            }
            res = static_cast<uint8_t>(*it); // static_cast, because `*it` can be signed
            // we need a byte count, because the iterator can reach the end before
            // we read the specified amount of bits
            auto byteCount = 1u;
            for (; nBytes > byteCount; ++byteCount) {
                if (end == ++it) break;
                res = (res << 8) + static_cast<uint8_t>(*it);
            }
            auto bitsRead = 8 * byteCount - curBit;
            if (end != it) {
                res >>= bitsLeft;
                bitsRead -= bitsLeft;
                curBit = (curBit + bitsRead) % 8;
                if (0 == curBit) {
                    ++it;
                }
            } else {
                curBit = 0;
            }
            res &= ResType(-1) >> (sizeof(ResType) * 8 - bitsRead);
            res |= preRead << bitsRead;
            return res;
        }

        template <unsigned MaxSize>
        auto ReadSigned(size_t n) -> decltype(MinSizeInt<MaxSize>())::type {
            using ResType  = decltype(MinSizeInt<MaxSize>())::type;
            using MaskType = decltype(MinSizeUInt<MaxSize>())::type;

            auto res = static_cast<ResType>(Read<MaxSize>(n));
            auto const mask = MaskType(1) << (n - 1);
            return (res ^ mask) - mask;
        }

        void Skip(unsigned n) {
            if (end == it) {
                return;
            }

            const auto tBits    = curBit + n;          // total amount of bits to read
                  auto nBytes   = (tBits - 1) / 8 + 1; // the amount of bytes that hold the bits we are reading
            const auto bitsLeft = nBytes * 8 - tBits;  // remaining bits belonging to the next read inside `nBytes`

            auto byteCount = 0u;
            for (byteCount = 1u; nBytes > byteCount; ++byteCount) {
                if (end == ++it) break; // avoid incrementing `byteCount` if we have reached the end
            }

            auto bitsRead = 8 * byteCount - curBit;

            if (end != it) {
                bitsRead -= bitsLeft;
                curBit = (curBit + bitsRead) % 8;
                if (0 == curBit) {
                    ++it;
                }
            } else {
                curBit = 0;
            }
        }

    private:
        InputIterator       it;
        const InputIterator end;
        uint8_t             curBit = 0;
    };
}
