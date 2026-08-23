#include <io/biniarchive.h>
#include <io/binoarchive.h>
#include <field.h>

template void Field<uint16_t>::load(io::BinIArchive<const char*>&, unsigned);
template void Field<uint16_t>::save(io::BinOArchive<uint8_t*>&, unsigned) const;
template void Field<float>::load(io::BinIArchive<const char*>&, unsigned);
template void Field<float>::save(io::BinOArchive<uint8_t*>&, unsigned) const;
