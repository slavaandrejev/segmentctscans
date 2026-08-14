#pragma once

#include <cstdint>
#include <cstdio>

#include <boost/iterator/iterator_facade.hpp>

namespace io
{
    class FileByteIterator :
        public boost::iterator_facade<
            FileByteIterator
          , uint8_t
          , boost::forward_traversal_tag
          , uint8_t
        >
    {
        friend class boost::iterator_core_access;

    public:
        FileByteIterator()
        {}

        FileByteIterator(FILE *inFile)
          : inFile(inFile)
        {
            increment();
        }

    private:
        uint8_t dereference() const {
            return currentValue;
        }

        bool equal(const FileByteIterator& rhs) const {
            return inFile == rhs.inFile;
        }

        void increment() {
            if (nullptr != inFile) {
                int c = fgetc(inFile);
                if (EOF == c) {
                    inFile = nullptr;
                }
                currentValue = static_cast<uint8_t>(c);
            }
        }

        uint8_t currentValue = 0;
        FILE   *inFile       = nullptr;
    };
}