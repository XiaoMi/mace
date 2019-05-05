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
#ifndef MACE_OPS_OPENCL_IMAGE_CROP_H_
#define MACE_OPS_OPENCL_IMAGE_CROP_H_

#include "mace/ops/opencl/crop.h"

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
class CropKernel : public OpenCLCropKernel {
 public:
  explicit CropKernel(
      const std::vector<int> &offset)
      : offset_(offset) {}
  MaceStatus Compute(
      OpContext *context,
      const std::vector<const Tensor *> &input_list,
      Tensor *output) override;

 private:
  std::vector<int> offset_;
  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus CropKernel<T>::Compute(
    OpContext *context,
    const std::vector<const Tensor *> &input_list,
    Tensor *output) {
  const int32_t inputs_count = static_cast<int32_t>(input_list.size());
  MACE_CHECK(inputs_count >= 2)
    << "Crop opencl kernel only support 2 elements input";
  const Tensor *input0 = input_list[0];
  const Tensor *input1 = input_list[1];
  const uint32_t in0_dims = static_cast<uint32_t >(input0->dim_size());
  const uint32_t in1_dims = static_cast<uint32_t >(input0->dim_size());
  MACE_CHECK(in0_dims == 4 && in1_dims == 4,
             "Crop op only supports 4-dims inputs now.");

  std::vector<int32_t> offsets(4, 0);

  std::vector<index_t> output_shape(input0->shape());
  for (index_t i = 0; i < in0_dims; ++i) {
    if (offset_[i] >= 0) {
      output_shape[i] = input1->dim(i);
      offsets[i] = offset_[i];
      MACE_CHECK(input0->dim(i) - offset_[i] >= input1->dim(i))
        << "the crop for dimension " << i
        << " is out of bound, first input size "
        << input0->dim(i) << ", offset " << offsets[i]
        << ", second input size " << input1->dim(i);
    }
  }
  MACE_CHECK(offsets[3] % 4 == 0,
             "MACE opencl only supports cropping channel"
                 " offset divisible by 4.");
  std::vector<size_t> image_shape;
  OpenCLUtil::CalImage2DShape(output_shape,
                              OpenCLBufferType::IN_OUT_CHANNEL,
                              &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  const index_t offset_chan_blk = RoundUpDiv4(offsets[3]);
  const index_t channel_blk = RoundUpDiv4(output->dim(3));
  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk), static_cast<uint32_t>(output->dim(2)),
      static_cast<uint32_t>(output->dim(0) * output->dim(1))
  };

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("crop");
    built_options.emplace("-Dcrop=" + kernel_name);
    auto dt = DataTypeToEnum<T>::value;
    built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToCLCMDDt(dt));
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("crop", kernel_name,
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
    kernel_.setArg(idx++, static_cast<int>(offsets[0]));
    kernel_.setArg(idx++, static_cast<int>(offsets[1]));
    kernel_.setArg(idx++, static_cast<int>(offsets[2]));
    kernel_.setArg(idx++, static_cast<int>(offset_chan_blk));
    kernel_.setArg(idx++, static_cast<int>(input0->dim(1)));
    kernel_.setArg(idx++, static_cast<int>(input0->dim(2)));
    kernel_.setArg(idx++, static_cast<int>(output->dim(1)));
    kernel_.setArg(idx++, static_cast<int>(output->dim(2)));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input0->shape();
  }

  const std::vector<uint32_t> lws = Default3DLocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("crop_opencl_kernel", output->dim(0), output->dim(1),
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

#endif  // MACE_OPS_OPENCL_IMAGE_CROP_H_
