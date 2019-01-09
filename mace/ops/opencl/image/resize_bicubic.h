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

#include "mace/core/op_context.h"
#include "mace/core/tensor.h"
#include "mace/ops/opencl/helper.h"
#include "mace/ops/resize_bicubic.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {
namespace resize_bicubic {
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

template <typename T>
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

template <typename T>
MaceStatus ResizeBicubicKernel<T>::Compute(
    OpContext *context,
    const Tensor *input,
    Tensor *output) {
  const index_t batch = input->dim(0);
  const index_t in_height = input->dim(1);
  const index_t in_width = input->dim(2);
  const index_t channels = input->dim(3);

  const index_t channel_blocks = RoundUpDiv4(channels);
  const index_t out_height = out_height_;
  const index_t out_width = out_width_;

  const uint32_t gws[3] = {static_cast<uint32_t>(channel_blocks),
                           static_cast<uint32_t>(out_width),
                           static_cast<uint32_t>(out_height * batch)};

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();
  MACE_OUT_OF_RANGE_DEFINITION;

  if (kernel_.get() == nullptr) {
    auto dt = DataTypeToEnum<T>::value;
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("resize_bicubic_nocache");
    built_options.emplace("-Dresize_bicubic_nocache=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    built_options.emplace(
        MakeString("-DTABLE_SIZE=",
                   mace::ops::resize_bicubic::kTableSize));
    MACE_RETURN_IF_ERROR(
        runtime->BuildKernel("resize_bicubic",
                             kernel_name,
                             built_options,
                             &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  MACE_OUT_OF_RANGE_INIT(kernel_);
  if (!IsVecEqual(input_shape_, input->shape())) {
    MACE_CHECK(out_height > 0 && out_width > 0);
    std::vector<index_t> output_shape{batch, out_height, out_width, channels};

    std::vector<size_t> output_image_shape;
    OpenCLUtil::CalImage2DShape(output_shape, OpenCLBufferType::IN_OUT_CHANNEL,
                                &output_image_shape);
    MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, output_image_shape));

    float height_scale =
        mace::ops::resize_bicubic::CalculateResizeScale(
            in_height, out_height, align_corners_);
    float width_scale =
        mace::ops::resize_bicubic::CalculateResizeScale(
            in_width, out_width, align_corners_);

    uint32_t idx = 0;
    MACE_OUT_OF_RANGE_SET_ARGS(kernel_);
    MACE_SET_3D_GWS_ARGS(kernel_, gws);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));
    kernel_.setArg(idx++, height_scale);
    kernel_.setArg(idx++, width_scale);
    kernel_.setArg(idx++, static_cast<int32_t>(in_height));
    kernel_.setArg(idx++, static_cast<int32_t>(in_width));
    kernel_.setArg(idx++, static_cast<int32_t>(out_height));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t>
      lws = resize_bicubic::LocalWS(runtime, gws, kwg_size_);
  std::string tuning_key =
      Concat("resize_bicubic_opencl_kernel", output->dim(0), output->dim(1),
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

#endif  // MACE_OPS_OPENCL_IMAGE_RESIZE_BICUBIC_H_
