//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_TYPES_H_
#define MACE_CORE_TYPES_H_

#include <cstdint>
#include <string>

#include "mace/public/mace.h"
#include "include/half.hpp"

namespace mace {

typedef int64_t index_t;

using half = half_float::half;

bool DataTypeCanUseMemcpy(DataType dt);

size_t GetEnumTypeSize(const DataType dt);

std::string DataTypeToString(const DataType dt);

template <class T>
struct IsValidDataType;

template <class T>
struct DataTypeToEnum {
  static_assert(IsValidDataType<T>::value, "Specified Data Type not supported");
};

// EnumToDataType<VALUE>::Type is the type for DataType constant VALUE, e.g.
// EnumToDataType<DT_FLOAT>::Type is float.
template <DataType VALUE>
struct EnumToDataType {};  // Specializations below

// Template specialization for both DataTypeToEnum and EnumToDataType.
#define MATCH_TYPE_AND_ENUM(TYPE, ENUM)     \
  template <>                               \
  struct DataTypeToEnum<TYPE> {             \
    static DataType v() { return ENUM; }    \
    static constexpr DataType value = ENUM; \
  };                                        \
  template <>                               \
  struct IsValidDataType<TYPE> {            \
    static constexpr bool value = true;     \
  };                                        \
  template <>                               \
  struct EnumToDataType<ENUM> {             \
    typedef TYPE Type;                      \
  }

MATCH_TYPE_AND_ENUM(half, DT_HALF);
MATCH_TYPE_AND_ENUM(float, DT_FLOAT);
MATCH_TYPE_AND_ENUM(double, DT_DOUBLE);
MATCH_TYPE_AND_ENUM(int32_t, DT_INT32);
MATCH_TYPE_AND_ENUM(uint16_t, DT_UINT16);
MATCH_TYPE_AND_ENUM(uint8_t, DT_UINT8);
MATCH_TYPE_AND_ENUM(int16_t, DT_INT16);
MATCH_TYPE_AND_ENUM(int8_t, DT_INT8);
MATCH_TYPE_AND_ENUM(std::string, DT_STRING);
MATCH_TYPE_AND_ENUM(int64_t, DT_INT64);
MATCH_TYPE_AND_ENUM(uint32_t, DT_UINT32);
MATCH_TYPE_AND_ENUM(bool, DT_BOOL);

static const int32_t kint32_tmax = ((int32_t)0x7FFFFFFF);
}  // namespace mace

#endif  // MACE_CORE_TYPES_H_
