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
#ifndef MACE_OPS_OPENCL_IMAGE_SPACE_TO_BATCH_H_
#define MACE_OPS_OPENCL_IMAGE_SPACE_TO_BATCH_H_

#include "mace/ops/opencl/space_to_batch.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class SpaceToBatchKernel : public OpenCLSpaceToBatchKernel {
 public:
  MaceStatus Compute(
      OpContext *context,
      const Tensor *space_tensor,
      const std::vector<int> &paddings,
      const std::vector<int> &block_shape,
      const std::vector<index_t> &output_shape,
      Tensor *batch_tensor) override;

 private:
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus SpaceToBatchKernel<T>::Compute(
    OpContext *context,
    const Tensor *space_tensor,
    const std::vector<int> &paddings,
    const std::vector<int> &block_shape,
    const std::vector<index_t> &output_shape,
    Tensor *batch_tensor) {
  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(
      batch_tensor->ResizeImage(output_shape, output_image_shape));
  const char *kernel_name = "space_to_batch";
  const uint32_t chan_blk = RoundUpDiv4<uint32_t>(batch_tensor->dim(3));
  const uint32_t gws[3] = {
      chan_blk, static_cast<uint32_t>(batch_tensor->dim(2)),
      static_cast<uint32_t>(batch_tensor->dim(0) * batch_tensor->dim(1))};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::string obfuscated_kernel_name = MACE_OBFUSCATE_SYMBOL(kernel_name);
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::stringstream kernel_name_ss;
    kernel_name_ss << "-D" << kernel_name << "=" << obfuscated_kernel_name;
    built_options.emplace(kernel_name_ss.str());
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DataTypeToEnum<T>::value));
    built_options.emplace("-DCMD_DATA_TYPE=" +
        DtToCLCMDDt(DataTypeToEnum<T>::value));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("space_to_batch",
                                              obfuscated_kernel_name,
                                              built_options,
                                              &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, space_tensor->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);

    kernel_.setArg(idx++, *(space_tensor->opencl_image()));
    kernel_.setArg(idx++, *(batch_tensor->opencl_image()));
    kernel_.setArg(idx++, block_shape[0]);
    kernel_.setArg(idx++, block_shape[1]);
    kernel_.setArg(idx++, paddings[0]);
    kernel_.setArg(idx++, paddings[2]);
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(0)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(space_tensor->dim(2)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(1)));
    kernel_.setArg(idx++, static_cast<int32_t>(batch_tensor->dim(2)));

    input_shape_ = space_tensor->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat(kernel_name, batch_tensor->dim(0), batch_tensor->dim(1),
             batch_tensor->dim(2), batch_tensor->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_SPACE_TO_BATCH_H_
