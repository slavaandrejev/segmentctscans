#pragma once

#include <cstdint>
#include <cstdio>
#include <memory>

#include <boost/iterator/iterator_facade.hpp>

namespace io
{
    class FileByteOutIterator :
        public boost::iterator_facade<
            FileByteOutIterator
          , uint8_t
          , boost::forward_traversal_tag
          >
    {
        friend class boost::iterator_core_access;
    public:
        // requirement of std::output_iterator
        FileByteOutIterator()
        {}

        explicit FileByteOutIterator(FILE *file)
          : file(file)
          , filePos(std::make_unique<off_t>(0).release())
          , itPos(0)
          , curValue(0)
          , needSaving(false)
        {}

        FileByteOutIterator(const FileByteOutIterator &cp)
          : file(cp.file)
          , filePos(cp.filePos)
          , itPos(cp.itPos)
          , curValue(cp.curValue)
          , needSaving(false)
        {}

        ~FileByteOutIterator() {
            if (needSaving) {
                SaveCurValue();
            }
        }

        reference dereference() const {
            needSaving = true;
            return curValue;
        }

        void increment() {
            if (needSaving) {
                SaveCurValue();
                needSaving = false;
            } else {
                fseek(file, 1, SEEK_CUR);
                (*filePos)++;
            }
            ++itPos;
        }

    private:
        void SaveCurValue() {
            if (*filePos != itPos) {
                fseek(file, itPos, SEEK_SET);
                *filePos = itPos;
            }
            fputc(curValue, file);
            (*filePos)++;
        }

        FILE                 *file = nullptr;
        std::shared_ptr<long> filePos;
        long                  itPos;
        mutable uint8_t       curValue;
        mutable bool          needSaving;
    };
}