// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MICRO_BASE_TYPES_H_
#define MICRO_BASE_TYPES_H_

#include "micro/include/public/micro.h"
#include "micro/include/utils/bfloat16.h"

namespace micro {

#ifdef MACE_ENABLE_BFLOAT16
typedef BFloat16 mifloat;
#else
typedef float mifloat;
#endif  // MACE_ENABLE_BFLOAT16

template<class T>
struct DataTypeToEnum;

template<DataType VALUE>
struct EnumToDataType;

#ifndef MACE_MAPPING_DATA_TYPE_AND_ENUM
#define MACE_MAPPING_DATA_TYPE_AND_ENUM(DATA_TYPE, ENUM_VALUE)  \
  template <>                                                   \
  struct DataTypeToEnum<DATA_TYPE> {                            \
    static DataType v() { return ENUM_VALUE; }                  \
    static const DataType value = ENUM_VALUE;                   \
  };                                                            \
  template <>                                                   \
  struct EnumToDataType<ENUM_VALUE> {                           \
    typedef DATA_TYPE Type;                                     \
  };
#endif  // MACE_MAPPING_DATA_TYPE_AND_ENUM

MACE_MAPPING_DATA_TYPE_AND_ENUM(float, DT_FLOAT);
MACE_MAPPING_DATA_TYPE_AND_ENUM(uint8_t, DT_UINT8);
MACE_MAPPING_DATA_TYPE_AND_ENUM(int32_t, DT_INT32);
#ifdef MACE_ENABLE_BFLOAT16
MACE_MAPPING_DATA_TYPE_AND_ENUM(BFloat16, DT_BFLOAT16);
#endif

struct QuantizeInfo {
  float scale;
  int32_t zero;
};

namespace ops {
namespace eltwise {  // for redefine

enum Type {
  SUM = 0,
  SUB = 1,
  PROD = 2,
  DIV = 3,
  MIN = 4,
  MAX = 5,
  NEG = 6,
  ABS = 7,
  SQR_DIFF = 8,
  POW = 9,
  EQUAL = 10,
  FLOOR_DIV = 11,
  CLIP = 12,
  SIGN = 13,
  NONE = 14,
};

}  // namespace eltwise
}  // namespace ops

}  // namespace micro

#endif  // MICRO_BASE_TYPES_H_
