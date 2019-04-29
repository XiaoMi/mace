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

#include <cstdint>
#include <map>

#include "mace/core/types.h"
#include "mace/utils/logging.h"

namespace mace {

bool DataTypeCanUseMemcpy(DataType dt) {
  switch (dt) {
    case DT_FLOAT:
    case DT_UINT8:
    case DT_INT32:
      return true;
    default:
      return false;
  }
}

std::string DataTypeToString(const DataType dt) {
  static std::map<DataType, std::string> dtype_string_map = {
      {DT_FLOAT, "DT_FLOAT"},
      {DT_HALF, "DT_HALF"},
      {DT_UINT8, "DT_UINT8"},
      {DT_INT32, "DT_INT32"}};
  MACE_CHECK(dt != DT_INVALID, "Not support Invalid data type");
  return dtype_string_map[dt];
}

size_t GetEnumTypeSize(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return sizeof(float);
    case DT_HALF:
      return sizeof(half);
#if defined(MACE_ENABLE_NEON) && defined(__ANDROID__)
    case DT_FLOAT16:
      return sizeof(float16_t);
#endif
    case DT_UINT8:
      return sizeof(uint8_t);
    case DT_INT32:
      return sizeof(int32_t);
    default:
      LOG(FATAL) << "Unsupported data type: " << dt;
      return 0;
  }
}

}  // namespace mace
