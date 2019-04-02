// Copyright 2018 The MACE Authors. All Rights Reserved.
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
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
#include <arm_neon.h>
#endif

#include "mace/proto/mace.pb.h"
#include "include/half.hpp"

namespace mace {

typedef int64_t index_t;

using half = half_float::half;

bool DataTypeCanUseMemcpy(DataType dt);

size_t GetEnumTypeSize(const DataType dt);

std::string DataTypeToString(const DataType dt);

template <class T>
struct DataTypeToEnum;

template <DataType VALUE>
struct EnumToDataType;

#define MACE_MAPPING_DATA_TYPE_AND_ENUM(DATA_TYPE, ENUM_VALUE)  \
  template <>                                                   \
  struct DataTypeToEnum<DATA_TYPE> {                            \
    static DataType v() { return ENUM_VALUE; }                  \
    static constexpr DataType value = ENUM_VALUE;               \
  };                                                            \
  template <>                                                   \
  struct EnumToDataType<ENUM_VALUE> {                           \
    typedef DATA_TYPE Type;                                     \
  };

MACE_MAPPING_DATA_TYPE_AND_ENUM(half, DT_HALF);
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
MACE_MAPPING_DATA_TYPE_AND_ENUM(float16_t, DT_FLOAT16);
#endif
MACE_MAPPING_DATA_TYPE_AND_ENUM(float, DT_FLOAT);
MACE_MAPPING_DATA_TYPE_AND_ENUM(uint8_t, DT_UINT8);
MACE_MAPPING_DATA_TYPE_AND_ENUM(int32_t, DT_INT32);

enum FrameworkType {
  TENSORFLOW = 0,
  CAFFE = 1,
};

}  // namespace mace

#endif  // MACE_CORE_TYPES_H_
