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
#ifndef MACE_OPS_OPENCL_IMAGE_RESIZE_BICUBIC_H_
#define MACE_OPS_OPENCL_IMAGE_RESIZE_BICUBIC_H_

#include "mace/ops/opencl/resize_bicubic.h"

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {
namespace resize_bicubic {
constexpr int64_t kTableSize = (1u << 10);

inline std::vector<uint32_t> LocalWS(OpenCLRuntime *runtime,
                                     const uint32_t *gws,
                                     const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  uint64_t cache_size = runtime->device_global_mem_cache_size();
  uint32_t base = std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
  lws[1] = std::min<uint32_t>(gws[1], kwg_size);
  if (lws[1] >= base) {
    lws[0] = std::min<uint32_t>(gws[0], base);
  } else {
    lws[0] = gws[0] / 8;
    if (lws[0] == 0) {
      lws[0] = gws[0];
    }
  }
  lws[0] = std::min<uint32_t>(lws[0], kwg_size / lws[1]);
  const uint32_t lws_size = lws[0] * lws[1];
  lws[2] = gws[2] / 8;
  if (lws[2] == 0) {
    lws[2] = gws[2];
  }
  lws[2] = std::max<uint32_t>(std::min<uint32_t>(lws[2], kwg_size / lws_size),
                              1);
  return lws;
}

}  // namespace resize_bicubic

class ResizeBicubicKernel : public OpenCLResizeBicubicKernel {
 public:
  ResizeBicubicKernel(bool align_corners,
                      const index_t out_height,
                      const index_t out_width)
      : align_corners_(align_corners),
        out_height_(out_height),
        out_width_(out_width) {}

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  bool align_corners_;
  index_t out_height_;
  index_t out_width_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_RESIZE_BICUBIC_H_
