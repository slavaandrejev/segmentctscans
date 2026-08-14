#pragma once

#include <boost/archive/detail/common_iarchive.hpp>
#include <boost/archive/detail/interface_iarchive.hpp>
#include <boost/archive/detail/iserializer.hpp>

#include "io/readprimitives.h"

namespace io
{
    template <typename Archive>
    class BasicBinIArchive
      : public boost::archive::detail::common_iarchive<Archive>
    {
        friend class boost::archive::detail::interface_iarchive<Archive>;
        friend class boost::archive::load_access;

        using base = boost::archive::detail::common_iarchive<Archive>;
    protected:
        BasicBinIArchive()
          : base(0)
        {}

        template <typename T>
        void load_override(T &t) {
            this->base::load_override(t);
        }

        void load_override(boost::archive::version_type &) {}
        void load_override(boost::archive::object_id_type &) {}
        void load_override(boost::archive::class_id_type &) {}
        void load_override(boost::archive::class_id_optional_type &) {}
        void load_override(boost::archive::class_name_type &) {}
        void load_override(boost::archive::tracking_type &) {}
    };

    // Adds serialization of primitive types.
    template <typename Archive, typename InputIterator>
    class BinIArchiveImpl
      : public ReadPrimitivesFromBin<InputIterator>
      , public BasicBinIArchive<Archive>
    {
        friend class boost::archive::load_access;
    public:
        template <typename It>
        BinIArchiveImpl(It &&start, It &&finish)
          : ReadPrimitivesFromBin<InputIterator>(
                std::forward<It>(start)
              , std::forward<It>(finish))
        {}
    };

    // Removes `Archive` from the template parameters.
    template <typename InputIterator>
    class BinIArchive
      : public BinIArchiveImpl<BinIArchive<InputIterator>, InputIterator>
    {
        using Base = BinIArchiveImpl<BinIArchive<InputIterator>, InputIterator>;
    public:
        using Base::Base;
    };

    template <typename InputIterator>
    BinIArchive(InputIterator, InputIterator) -> BinIArchive<InputIterator>;
}
