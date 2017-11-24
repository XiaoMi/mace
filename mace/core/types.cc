//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/types.h"

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


size_t GetEnumTypeSize(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return sizeof(float);
    case DT_HALF:
      return sizeof(half);
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

}  //  namespace mace