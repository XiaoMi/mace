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

#include <cstdint>
#include <map>

#include "mace/core/types.h"
#include "mace/utils/logging.h"

namespace mace {

bool DataTypeCanUseMemcpy(DataType dt) {
  switch (dt) {
    case DT_FLOAT:
    case DT_DOUBLE:
    case DT_INT32:
    case DT_INT64:
    case DT_UINT32:
    case DT_UINT16:
    case DT_UINT8:
    case DT_INT16:
    case DT_INT8:
    case DT_BOOL:
      return true;
    default:
      return false;
  }
}

std::string DataTypeToString(const DataType dt) {
  static std::map<DataType, std::string> dtype_string_map = {
      {DT_FLOAT, "DT_FLOAT"},
#ifdef MACE_ENABLE_OPENCL
      {DT_HALF, "DT_HALF"},
#endif
      {DT_DOUBLE, "DT_DOUBLE"},
      {DT_UINT8, "DT_UINT8"},
      {DT_INT8, "DT_INT8"},
      {DT_INT32, "DT_INT32"},
      {DT_UINT32, "DT_UINT32"},
      {DT_UINT16, "DT_UINT16"},
      {DT_INT64, "DT_INT64"},
      {DT_BOOL, "DT_BOOL"},
      {DT_STRING, "DT_STRING"}};
  MACE_CHECK(dt != DT_INVALID) << "Not support Invalid data type";
  return dtype_string_map[dt];
}

size_t GetEnumTypeSize(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return sizeof(float);
#ifdef MACE_ENABLE_OPENCL
    case DT_HALF:
      return sizeof(half);
#endif
    case DT_UINT8:
      return sizeof(uint8_t);
    case DT_INT8:
      return sizeof(int8_t);
    case DT_DOUBLE:
      return sizeof(double);
    case DT_INT32:
      return sizeof(int32_t);
    case DT_UINT32:
      return sizeof(uint32_t);
    case DT_UINT16:
      return sizeof(uint16_t);
    case DT_INT16:
      return sizeof(int16_t);
    case DT_INT64:
      return sizeof(int64_t);
    default:
      LOG(FATAL) << "Unsupported data type";
      return 0;
  }
}

}  // namespace mace
