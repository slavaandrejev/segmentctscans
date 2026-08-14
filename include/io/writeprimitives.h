#pragma once

#include <concepts>
#include <cstdint>
#include <iterator>
#include <utility>

#include <x86intrin.h>

#include <boost/archive/detail/oserializer.hpp>

#include "io/outbitstream.h"
#include "io/primitivetypes.h"

namespace io
{
    template <typename OutputIterator>
        requires std::output_iterator<OutputIterator, std::iter_value_t<OutputIterator>>
    class WritePrimitivesToBin
    {
        friend class boost::archive::save_access;

    public:
        template <typename It>
        WritePrimitivesToBin(It &&it)
          : os(std::forward<It>(it))
        {}

        void save(const int &v) {
            save(se(v));
        }

        void save(const b &v) {
            os.template Write<8>(v);
        }

        template <unsigned N, unsigned Value>
        void save(const f<N, Value> &) {
            os.template Write<N>(Value);
        }

        template <unsigned N>
        void save(const u<N> &v) {
            os.template Write<N>(v);
        }

        void save(const u<1> &v) {
            os.template Write<1>(v ? 1 : 0);
        }

        void save(const uv &v) {
            os.template Write<32>(v, v.Bits());
        }

        void save(const ue &v) {
            // Exponential-Golomb coding
            auto numBits = 64 - __builtin_clzll(uint64_t{v + 1});
            os.template Write<64>(0, numBits - 1);
            os.template Write<64>(v + 1, numBits);
        }

        void save(const se &v) {
            auto av = ue(2 * std::abs(v));
            if (v > 0) --av;
            save(av);
        }

        const auto& Os() const {
            return os;
        }

        size_t size() const {
            return os.size();
        }

    private:
        io::OutputBitStream<OutputIterator> os;
    };
}
