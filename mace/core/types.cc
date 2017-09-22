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

} //  namespace mace