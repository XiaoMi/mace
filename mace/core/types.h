// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_CORE_TYPES_H_
#define MACE_CORE_TYPES_H_

#include <cstdint>
#include <string>

#include "mace/public/mace_types.h"
#ifdef MACE_ENABLE_OPENCL
#include "include/half.hpp"
#endif

namespace mace {

typedef int64_t index_t;

#ifdef MACE_ENABLE_OPENCL
using half = half_float::half;
#endif

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

#ifdef MACE_ENABLE_OPENCL
MATCH_TYPE_AND_ENUM(half, DT_HALF);
#endif
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
