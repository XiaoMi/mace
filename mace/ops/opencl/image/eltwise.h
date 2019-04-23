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
#ifndef MACE_OPS_OPENCL_IMAGE_ELTWISE_H_
#define MACE_OPS_OPENCL_IMAGE_ELTWISE_H_

#include "mace/ops/opencl/eltwise.h"

#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <string>

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/eltwise.h"
#include "mace/ops/opencl/helper.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

template <typename T>
class EltwiseKernel : public OpenCLEltwiseKernel {
 public:
  explicit EltwiseKernel(
      const EltwiseType type,
      const std::vector<float> &coeff,
      const float scalar_input,
      const int32_t scalar_input_index)
      : type_(type),
        coeff_(coeff),
        scalar_input_(scalar_input),
        scalar_input_index_(scalar_input_index) {}
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input0,
      const Tensor *input1,
      Tensor *output) override;

 private:
  EltwiseType type_;
  std::vector<float> coeff_;
  float scalar_input_;
  int32_t scalar_input_index_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus EltwiseKernel<T>::Compute(
    OpContext *context,
    const Tensor *input0,
    const Tensor *input1,
    Tensor *output) {
  bool swapped = false;
  std::string input1_type = "";
  if (input1 == nullptr) {
    input1_type = "INPUT_SCALAR";
  } else {
    MACE_CHECK((input0->dim_size() == input1->dim_size()
        && input0->dim_size() == 4) ||
        input0->dim_size() == 1 || input1->dim_size() == 1)
      << "Inputs of Eltwise op must be same shape or fulfill broadcast logic";
    MACE_CHECK(type_ != EltwiseType::EQUAL)
      << "Eltwise op on GPU does not support EQUAL";
    // broadcast
    if (input0->size() != input1->size() ||
        input0->dim_size() != input1->dim_size()) {
      if (input0->size() < input1->size()
          || input0->dim_size() < input1->dim_size()) {
        std::swap(input0, input1);
        swapped = true;
      }
      if (input1->dim_size() == 1
          || (input1->dim(0) == 1 && input1->dim(1) == 1
              && input1->dim(2) == 1)) {
        // Tensor-Vector element wise
        if (input0->dim(3) == input1->dim(input1->dim_size()-1)) {
          input1_type = "INPUT_VECTOR";
        } else {
          LOG(FATAL) << "Inputs not match the broadcast logic, "
                     << MakeString(input0->shape()) << " vs "
                     << MakeString(input1->shape());
        }
      } else {  // must be 4-D
        if (input0->dim(0) == input1->dim(0)
            && input1->dim(1) == 1
            && input1->dim(2) == 1
            && input0->dim(3) == input1->dim(3)) {
          input1_type = "INPUT_BATCH_VECTOR";
        } else if (input0->dim(0) == input1->dim(0)
            && input0->dim(1) == input1->dim(1)
            && input0->dim(2) == input1->dim(2)
            && input1->dim(3) == 1) {
          // broadcast on channel dimension
          input1_type = "INPUT_TENSOR_BC_CHAN";
        } else {
          LOG(FATAL) << "Element-Wise op only support broadcast on"
                        " channel dimension:"
                        "Tensor-BatchVector(4D-[N,1,1,C]) "
                        "and Tensor-Tensor(4D-[N,H,W,1]). but got "
                     << MakeString(input0->shape()) << " vs "
                     << MakeString(input1->shape());
        }
      }
    }
  }

  if (scalar_input_index_ == 0) {
    swapped = !swapped;
  }

  std::vector<index_t> output_shape(4);
  output_shape[0] = input0->dim(0);
  output_shape[1] = input0->dim(1);
  output_shape[2] = input0->dim(2);
  output_shape[3] = input0->dim(3);

  std::vector<size_t> output_image_shape;
  OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                              &output_image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

  const index_t batch = output->dim(0);
  const index_t height = output->dim(1);
  const index_t width = output->dim(2);
  const index_t channels = output->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t batch_height_pixels = batch * height;

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(width),
                           static_cast<uint32_t>(batch_height_pixels)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("eltwise");
    built_options.emplace("-Deltwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    built_options.emplace(MakeString("-DELTWISE_TYPE=", type_));
    if (!input1_type.empty()) {
      built_options.emplace("-D" + input1_type);
    }
    if (swapped) built_options.emplace("-DSWAPPED");
    if (channels % 4 != 0) built_options.emplace("-DNOT_DIVISIBLE_FOUR");
    if (!coeff_.empty()) built_options.emplace("-DCOEFF_SUM");
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("eltwise", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input0->shape())) {
    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input0->opencl_image()));
    if (input1 == nullptr) {
      kernel_.setArg(idx++, scalar_input_);
    } else {
      kernel_.setArg(idx++, *(input1->opencl_image()));
    }
    kernel_.setArg(idx++, static_cast<int32_t>(height));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(channels));
    if (!coeff_.empty()) {
      kernel_.setArg(idx++, coeff_[0]);
      kernel_.setArg(idx++, coeff_[1]);
    }
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input0->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("eltwise_opencl_kernel", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, context->future()));
  MACE_OUT_OF_RANGE_VALIDATION;
  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_IMAGE_ELTWISE_H_
