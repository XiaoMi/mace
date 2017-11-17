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

std::string DataTypeToCLType(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return "float";
    case DT_HALF:
      return "half";
    case DT_UINT8:
      return "uchar";
    case DT_INT8:
      return "char";
    case DT_DOUBLE:
      return "double";
    case DT_INT32:
      return "int";
    case DT_UINT32:
      return "int";
    case DT_UINT16:
      return "ushort";
    case DT_INT16:
      return "short";
    case DT_INT64:
      return "long";
    default:
      LOG(FATAL) << "Unsupported data type";
      return "";
  }
}

}  //  namespace mace