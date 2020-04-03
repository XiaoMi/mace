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
#ifndef MACE_OPS_OPENCL_IMAGE_MVNORM_H_
#define MACE_OPS_OPENCL_IMAGE_MVNORM_H_

#include <memory>
#include <vector>

#include "mace/core/ops/op_context.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/core/runtime/opencl/opencl_helper.h"
#include "mace/ops/opencl/mvnorm.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

class MVNormKernel : public OpenCLMVNormKernel {
 public:
  explicit MVNormKernel(bool normalize_variance_,
                        bool across_channels, float eps);
  ~MVNormKernel() = default;

  MaceStatus Compute(
      OpContext *context, const Tensor *input, Tensor *output) override;

 private:
  void CheckImage(OpContext *context, const DataType dt,
                  const std::vector<index_t> &square_shape,
                  const std::vector<index_t> &mean_shape);
  MaceStatus ExecuteMeanNormKernel(OpContext *context,
                                   OpenCLRuntime *runtime,
                                   const uint32_t (&gws)[3],
                                   const Tensor *input,
                                   Tensor *output);
  MaceStatus ExecuteVarianceNormStep1Kernel(OpContext *context,
                                            OpenCLRuntime *runtime,
                                            const uint32_t (&gws)[3],
                                            const Tensor *input);
  MaceStatus ExecuteVarianceNormStep2Kernel(OpContext *context,
                                            OpenCLRuntime *runtime,
                                            const uint32_t (&gws)[3],
                                            const Tensor *input,
                                            Tensor *output);

 private:
  bool normalize_variance_;
  bool across_channels_;
  float eps_;

  cl::Kernel kernel_step1_;
  uint32_t kwg_size_step1_;
  cl::Kernel kernel_step2_;
  uint32_t kwg_size_step2_;

  // the cache of (X - EX)^2
  std::unique_ptr<Image> square_image_;
  // the cache of EX
  std::unique_ptr<Image> mean_image_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_MVNORM_H_
