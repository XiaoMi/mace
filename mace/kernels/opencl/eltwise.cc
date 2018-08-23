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

#include "mace/kernels/eltwise.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus EltwiseFunctor<DeviceType::GPU, T>::operator()(const Tensor *input0,
                                                          const Tensor *input1,
                                                          Tensor *output,
                                                          StatsFuture *future) {
  MACE_UNUSED(future);
  bool swapped = false;
  if (input1 != nullptr) {
    MACE_CHECK(input0->dim_size() == input1->dim_size() ||
               input0->dim_size() == 1 || input1->dim_size() == 1)
        << "Inputs of Eltwise op must be same shape";
    if (input0->size() != input1->size()) {
      if (input0->size() < input1->size()) {
        std::swap(input0, input1);
        swapped = true;
      }
      if (input1->dim_size() == 1) {
        MACE_CHECK(input0->dim(3) == input1->dim(0))
            << "Element-Wise op only support channel dimension broadcast";
      } else {
        MACE_CHECK((input0->dim(0) == input1->dim(0) || input1->dim(0) == 1) &&
                   input0->dim(3) == input1->dim(3) && input1->dim(1) == 1 &&
                   input1->dim(2) == 1)
            << "Element-Wise op only support channel dimension broadcast";
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
  CalImage2DShape(output_shape, BufferType::IN_OUT_CHANNEL,
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

  auto runtime = OpenCLRuntime::Global();
  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_);
    NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("eltwise");
    built_options.emplace("-Deltwise=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));
    built_options.emplace(MakeString("-DELTWISE_TYPE=", type_));
    if (input1 == nullptr) {
      built_options.emplace("-DINPUT_TYPE=1");
    } else if (input0->size() != input1->size()) {
      if (input1->dim(0) == 1 || input1->dim_size() == 1)
        built_options.emplace("-DINPUT_TYPE=3");
      else
        built_options.emplace("-DINPUT_TYPE=2");
      if (swapped) built_options.emplace("-DSWAPPED");
    }
    if (!coeff_.empty()) built_options.emplace("-DCOEFF_SUM");
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("eltwise", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }
  if (!IsVecEqual(input_shape_, input0->shape())) {
    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_3D_GWS_ARGS(kernel_);
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

  const std::vector<uint32_t> lws = Default3DLocalWS(gws, kwg_size_);
  std::string tuning_key =
      Concat("eltwise_opencl_kernel", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(kernel_, tuning_key,
                                           gws, lws, future));
  OUT_OF_RANGE_VALIDATION(kernel_error_);
  return MACE_SUCCESS;
}

template struct EltwiseFunctor<DeviceType::GPU, float>;
template struct EltwiseFunctor<DeviceType::GPU, half>;
}  // namespace kernels
}  // namespace mace
