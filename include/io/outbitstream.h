#pragma once

#include <concepts>
#include <cstdint>
#include <iterator>
#include <type_traits>

#include "minsizetype.h"

namespace io
{
    template <typename OutputIterator>
        requires (
            std::output_iterator<OutputIterator, std::iter_value_t<OutputIterator>>
         && std::integral<std::iter_value_t<OutputIterator>>
         && 1 == sizeof(std::iter_value_t<OutputIterator>))
    class OutputBitStream
    {
    public:
        explicit OutputBitStream(OutputIterator it)
          : it(it)
        {}

        template <unsigned MaxSize>
        void Write(decltype(MinSizeUInt<MaxSize>())::type what, unsigned numBits = MaxSize) {
            if (0 == numBits) return;
            what <<= (8 * sizeof(what) - numBits);
            if (0 == count) count = 1;
            for (;;) {
                if (8 == curBit) {
                    ++it;
                    ++count;
                    curBit = 0;
                }
                auto bitsLeftInByte = 8 - curBit;
                *it &= uint8_t(-1) << bitsLeftInByte;
                *it |= uint8_t(what >> (8 * sizeof(what) - bitsLeftInByte));
                auto bitsWillLeftInByte = int(bitsLeftInByte) - int(numBits);
                if (0 == (bitsWillLeftInByte - 7) / 8) { // negated how many byte boundaries we are crossing
                    curBit += numBits;
                    break;
                } else {
                    curBit   = 8;
                    numBits -= bitsLeftInByte;
                    what   <<= bitsLeftInByte;
                }
            }
        }

        size_t size() const {
            return count;
        }
    private:
        size_t count{};
        OutputIterator it;
        uint8_t        curBit = 0;
    };
}
