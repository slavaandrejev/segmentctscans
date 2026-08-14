#pragma once

#include <concepts>
#include <cstdint>
#include <utility>

#include <boost/archive/detail/iserializer.hpp>

#include "io/inbitstream.h"
#include "io/primitivetypes.h"

namespace io
{
    template <std::input_iterator InputIterator>
    class ReadPrimitivesFromBin
    {
        friend class boost::archive::load_access;

    public:
        template <typename It>
        ReadPrimitivesFromBin(It &&start, It &&finish)
          : is(std::forward<It>(start), std::forward<It>(finish))
        {}

        void load(int &v) {
            se sev;
            load(sev);
            v = sev;
        }

        void load(b &v) {
            v = is.template Read<8>();
        }

        template <unsigned N, unsigned Value>
        void load(f<N, Value> &) {
            is.template Read<N>();
        }

        template <unsigned N>
        void load(u<N> &v) {
            v = is.template Read<N>();
        }

        void load(u<1> &v) {
            v = 0 != is.template Read<1>();
        }

        void load(uv &v) {
            v = is.template Read<32>(v.Bits());
        }

        void load(ue &v) {
            auto zeroBits = 0;

            // Exponential-Golomb coding
            for (zeroBits = 0; 0 == is.template Read<1>(); ++zeroBits);
            if (0 == zeroBits) {
                v = 0;
            } else {
                v = (uint64_t{1} << zeroBits) + is.template Read<64>(zeroBits) - 1;
            }
        }

        void load(se &v) {
            auto ret = ue{};
            load(ret);

            if (0 == (ret & 1)) {
                ret >>= 1;
                v = 0 - ret;
            } else {
                v = (ret + 1) >> 1;
            }
        }

        const auto& Is() const {
            return is;
        }

    private:
        io::InputBitStream<InputIterator> is;
    };
}
