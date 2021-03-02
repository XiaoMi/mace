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

#ifndef MACE_RUNTIMES_OPENCL_CORE_OPENCL_UTIL_H_
#define MACE_RUNTIMES_OPENCL_CORE_OPENCL_UTIL_H_

#include <memory>
#include <string>
#include <vector>

#include "mace/core/types.h"
#include "mace/core/ops/ops_utils.h"
#include "mace/public/mace.h"
#include "mace/runtimes/opencl/core/cl2_header.h"

namespace mace {

class OpenCLUtil {
 public:
  static void CalImage2DShape(const std::vector<index_t> &shape, /* NHWC */
                              const BufferContentType type,
                              std::vector<size_t> *image_shape,
                              const int wino_blk_size = 2);

  static cl_channel_type DataTypeToCLChannelType(const DataType t);

  static void SetOpenclInputToCpuBuffer(OpConditionContext *context,
                                        size_t idx, DataType data_type);
};

}  // namespace mace
#endif  // MACE_RUNTIMES_OPENCL_CORE_OPENCL_UTIL_H_
