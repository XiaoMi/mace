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
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_helper.h"
#include "mace/runtimes/opencl/opencl_runtime.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/mvnorm.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

enum MeanType {
  SINGLE_CHANNEL,
  GROUP_CHANNELS,
  ACROSS_CHANNELS,
};

class MVNormKernel : public OpenCLMVNormKernel {
 public:
  explicit MVNormKernel(bool normalize_variance_, MeanType mean_type,
                        float eps, int group_num = 0);
  ~MVNormKernel() = default;

  MaceStatus Compute(
      OpContext *context, const Tensor *input, Tensor *output) override;

 private:
  MaceStatus DoCompute(OpContext *context, const Tensor *input,
                       Tensor *output, const index_t batch,
                       const index_t height, const index_t width,
                       const index_t channels, const index_t group_blocks);

  MaceStatus ExecuteMeanValueKernel(OpContext *context,
                                    OpenclExecutor *executor,
                                    const index_t batch,
                                    const index_t height,
                                    const index_t width,
                                    const index_t channel_blocks,
                                    const index_t group_blocks,
                                    const cl::Image *input_image,
                                    cl::Image *output_image);

  MaceStatus ExecuteMeanNormKernel(OpContext *context,
                                   OpenclExecutor *executor,
                                   const uint32_t (&gws)[3],
                                   const index_t height,
                                   const index_t group_blocks,
                                   const cl::Image *input,
                                   const cl::Image *mean_image,
                                   cl::Image *output);

  // compute the (X - EX)^2
  MaceStatus ExecuteVarianceNormStep1Kernel(OpContext *context,
                                            OpenclExecutor *executor,
                                            const uint32_t (&gws)[3],
                                            const index_t height,
                                            const index_t group_blocks,
                                            const cl::Image *input,
                                            const cl::Image *mean_image,
                                            cl::Image *output);

  // compute (X - EX) / (E((X - EX)^2)^0.5 + eps_)
  MaceStatus ExecuteVarianceNormStep2Kernel(OpContext *context,
                                            OpenclExecutor *executor,
                                            const uint32_t (&gws)[3],
                                            const index_t height,
                                            const index_t group_blocks,
                                            const cl::Image *input,
                                            const cl::Image *mean_image,
                                            const cl::Image *mean_image_sqr,
                                            cl::Image *output);

 private:
  const bool normalize_variance_;
  const MeanType mean_type_;
  const float eps_;
  const int group_num_;

  cl::Kernel kernel_mean_;
  uint32_t kwg_size_mean_;
  cl::Kernel kernel_step1_;
  uint32_t kwg_size_step1_;
  cl::Kernel kernel_step2_;
  uint32_t kwg_size_step2_;
};

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_MVNORM_H_
