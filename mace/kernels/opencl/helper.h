//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_OPENCL_HELPER_H_
#define MACE_KERNELS_OPENCL_HELPER_H_
#include "mace/core/types.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

enum BufferType {
  FILTER = 0,
  IN_OUT= 1,
  ARGUMENT = 2
};

void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                     const BufferType type,
                     std::vector<size_t> &image_shape);

std::string DtToCLCMDDt(const DataType dt);

std::string DtToUpstreamCLCMDDt(const DataType dt);

std::string DtToCLDt(const DataType dt);

std::string DtToUpstreamCLDt(const DataType dt);

}  // namespace kernels
} //  namespace mace
#endif //  MACE_KERNELS_OPENCL_HELPER_H_
