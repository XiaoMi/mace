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
#ifndef MACE_OPS_OPENCL_IMAGE_ONE_HOT_H_
#define MACE_OPS_OPENCL_IMAGE_ONE_HOT_H_

#include "mace/ops/opencl/one_hot.h"

#include <memory>
#include <vector>
#include <set>
#include <string>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"


namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class OneHotKernel : public OpenCLOneHotKernel {
 public:
  OneHotKernel(const int depth, const float on_value,
               const float off_value, const int axis)
      : depth_(depth), on_value_(on_value),
        off_value_(off_value), axis_(axis) {}

  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      Tensor *output) override;

 private:
  int depth_;
  float on_value_;
  float off_value_;
  int axis_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus OneHotKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {

  auto input_shape = input->shape();
  index_t axis = axis_ == -1 ? input->dim_size() : axis_;

  MACE_CHECK(input->dim_size() == 1, "OneHot GPU only supports 1D input");
  MACE_CHECK(axis >= 0 && axis <= input->dim_size());

  std::vector<index_t> output_shape =
      axis == 0 ? std::vector<index_t>{depth_, input_shape[0]} :
                  std::vector<index_t>{input_shape[0], depth_};
  std::vector<size_t> output_image_shape{
      static_cast<size_t>(RoundUpDiv4(output_shape[1])),
      static_cast<size_t>(output_shape[0])};
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("one_hot");
    built_options.emplace("-Done_hot=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    if (axis == 0) {
      built_options.emplace("-DAXIS_0");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("one_hot", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[2] = {
    static_cast<uint32_t>(output_image_shape[0]),
    static_cast<uint32_t>(output_image_shape[1])
  };
  MACE_OUT_OF_RANGE_INIT(kernel_);

  if (!IsVecEqual(input_shape_, input->shape())) {
    int idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_2D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
    if (axis == 0) {
      kernel_.setArg(idx++, static_cast<int>(input_shape[0]));
    }
    kernel_.setArg(idx++, on_value_);
    kernel_.setArg(idx++, off_value_);

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 64, 64, 0};
  std::string tuning_key = Concat("one_hot", output->dim(0), output->dim(1));
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));

  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_ONE_HOT_H_
