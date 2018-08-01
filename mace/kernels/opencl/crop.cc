// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include "mace/kernels/crop.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

namespace {
std::vector<uint32_t> LocalWS(const uint32_t *gws, const uint32_t kwg_size) {
  std::vector<uint32_t> lws(4, 0);
  if (kwg_size == 0) {
    lws[0] = lws[1] = lws[2] = 1;
  } else {
    uint64_t
        cache_size = OpenCLRuntime::Global()->device_global_mem_cache_size();
    uint32_t base = std::max<uint32_t>(cache_size / kBaseGPUMemCacheSize, 1);
    lws[1] = std::min<uint32_t>(gws[1], kwg_size);
    lws[0] = std::min<uint32_t>(base, kwg_size / lws[1]);
    const uint32_t lws_size = lws[0] * lws[1];
    lws[2] =
        std::max<uint32_t>(std::min<uint32_t>(base, kwg_size / lws_size), 1);
  }
  return lws;
}

}  // namespace


template <typename T>
MaceStatus CropFunctor<DeviceType::GPU, T>::operator()(
    const std::vector<const Tensor *> &input_list,
    Tensor *output,
    StatsFuture *future) {
  MACE_UNUSED(future);

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
  switch (axis_) {
    case 0:
      if (offset_.size() == 1) {
        offsets[0] = offset_[0];
        offsets[1] = offset_[0];
        offsets[2] = offset_[0];
        offsets[3] = offset_[0];
      } else if (offset_.size() == 4) {
        offsets[0] = offset_[0];
        offsets[1] = offset_[2];
        offsets[2] = offset_[3];
        offsets[3] = offset_[1];
      }
      for (int i = 0; i < 4; ++i) {
        output_shape[i] = input1->dim(i);
      }
      break;
    case 1:
      if (offset_.size() == 1) {
        offsets[1] = offset_[0];
        offsets[2] = offset_[0];
        offsets[3] = offset_[0];
      } else if (offset_.size() == 3) {
        offsets[1] = offset_[1];
        offsets[2] = offset_[2];
        offsets[3] = offset_[0];
      }
      for (int i = 1; i < 4; ++i) {
        output_shape[i] = input1->dim(i);
      }
      break;
    case 2:
      if (offset_.size() == 1) {
        offsets[1] = offset_[0];
        offsets[2] = offset_[0];
      } else if (offset_.size() == 2) {
        offsets[1] = offset_[0];
        offsets[2] = offset_[1];
      }
      output_shape[1] = input1->dim(1);
      output_shape[2] = input1->dim(2);
      break;
    case 3:
      if (offset_.size() == 1) {
        offsets[2] = offset_[0];
      }
      output_shape[2] = input1->dim(2);
      break;
    default:
      MACE_CHECK(axis_ >= 0 && axis_ < 4, "axis is out of boundary.");
      break;
  }
  MACE_CHECK(offsets[3] % 4 == 0,
      "MACE opencl only supports cropping channel offset divisible by 4.");
  for (index_t i = 0; i < 4; ++i) {
    MACE_CHECK(input0->dim(i) - offsets[i] >= input1->dim(i))
        << "the crop for dimension" << i << "is out of bound with size"
        << input1->dim(i) << "and offset" << offsets[i];
  }
  std::vector<size_t> image_shape;
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL, &image_shape);
  MACE_RETURN_IF_ERROR(output->ResizeImage(output_shape, image_shape));

  const index_t offset_chan_blk = RoundUpDiv4(offsets[3]);
  const index_t channel_blk = RoundUpDiv4(output->dim(3));
  const uint32_t gws[3] = {
      static_cast<uint32_t>(channel_blk), static_cast<uint32_t>(output->dim(2)),
      static_cast<uint32_t>(output->dim(0) * output->dim(1))
  };

  auto runtime = OpenCLRuntime::Global();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
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
  if (!IsVecEqual(input_shape_, input0->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
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

  const std::vector<uint32_t> lws = LocalWS(gws, kwg_size_);
  std::string tuning_key =
      Concat("crop_opencl_kernel", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(kernel_, tuning_key,
                                           gws, lws, future));
  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct CropFunctor<DeviceType::GPU, float>;
template struct CropFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
