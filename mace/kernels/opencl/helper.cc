//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/kernels/opencl/helper.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

// [(c+3)/4*W, N * H]
void CalInOutputImageShape(const std::vector<index_t> &shape, /* NHWC */
                        std::vector<size_t> &image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape.resize(2);
  image_shape[0] = RoundUpDiv4(shape[3]) * shape[2];
  image_shape[1] = shape[0] * shape[1];
}

// [H * W * 4, (Oc + 3) / 4]
void CalFilterImageShape(const std::vector<index_t> &shape, /* HWIO*/
                         std::vector<size_t> &image_shape) {
  MACE_CHECK(shape.size() == 4);
  image_shape.resize(2);
  image_shape[0] = shape[0] * shape[1] * RoundUp<index_t>(shape[2], 4);
  image_shape[1] = RoundUpDiv4(shape.back());
}

// [(size + 3) / 4, 1]
void CalArgImageShape(const std::vector<index_t> &shape,
                      std::vector<size_t> &image_shape) {
  MACE_CHECK(shape.size() == 1);
  image_shape.resize(2);
  image_shape[0] = RoundUpDiv4(shape[0]);
  image_shape[1] = 1;
}

void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                     const BufferType type,
                     std::vector<size_t> &image_shape) {
  switch (type) {
    case FILTER:
      CalFilterImageShape(shape, image_shape);
      break;
    case IN_OUT:
      CalInOutputImageShape(shape, image_shape);
      break;
    case ARGUMENT:
      CalArgImageShape(shape, image_shape);
      break;
    default:
      LOG(FATAL) << "Mace not supported yet.";
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

std::string DataTypeToOpenclCMDDataType(const DataType dt) {
  switch (dt) {
    case DT_FLOAT:
      return "f";
    case DT_HALF:
      return "h";
    default:
      LOG(FATAL) << "Not supported data type for opencl cmd data type";
      return "";
  }
}

}  // namespace kernels
}  // namespace mace
