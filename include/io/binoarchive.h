#pragma once

#include <boost/archive/detail/common_oarchive.hpp>
#include <boost/archive/detail/interface_oarchive.hpp>
#include <boost/archive/detail/oserializer.hpp>

#include "io/writeprimitives.h"

namespace io
{
    template <typename Archive>
    class BasicBinOArchive
      : public boost::archive::detail::common_oarchive<Archive>
    {
        friend class boost::archive::detail::interface_oarchive<Archive>;
        friend class boost::archive::save_access;

        using base = boost::archive::detail::common_oarchive<Archive>;
    protected:
        BasicBinOArchive()
          : base(0)
        {}

        template <typename T>
        void save_override(const T &t) {
            this->base::save_override(t);
        }

        void save_override(const boost::archive::version_type &) {}
        void save_override(const boost::archive::object_id_type &) {}
        void save_override(const boost::archive::object_reference_type &) {}
        void save_override(const boost::archive::class_id_type &) {}
        void save_override(const boost::archive::class_id_optional_type &) {}
        void save_override(const boost::archive::class_id_reference_type &) {}
        void save_override(const boost::archive::class_name_type &) {}
        void save_override(const boost::archive::tracking_type &) {}
    };

    // Adds serialization of primitive types.
    template <typename Archive, typename OutputIterator>
    class BinOArchiveImpl
      : public WritePrimitivesToBin<OutputIterator>
      , public BasicBinOArchive<Archive>
    {
        friend class boost::archive::save_access;
    public:
        template <typename It>
        BinOArchiveImpl(It &&it)
          : WritePrimitivesToBin<OutputIterator>(std::forward<It>(it))
        {}

    };

    // Removes `Archive` from the template parameters.
    template <typename OutputIterator>
    class BinOArchive
      : public BinOArchiveImpl<BinOArchive<OutputIterator>, OutputIterator>
    {
        using Base = BinOArchiveImpl<BinOArchive<OutputIterator>, OutputIterator>;
    public:
        using Base::Base;
    };

    template <typename OutputIterator>
    BinOArchive(OutputIterator) -> BinOArchive<OutputIterator>;
}
