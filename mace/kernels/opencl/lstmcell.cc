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

#include "mace/kernels/lstmcell.h"
#include "mace/core/runtime/opencl/opencl_runtime.h"
#include "mace/kernels/opencl/helper.h"
#include "mace/utils/tuner.h"
#include "mace/utils/utils.h"

namespace mace {
namespace kernels {

template <typename T>
MaceStatus LSTMCellFunctor<DeviceType::GPU, T>::operator()(
    const Tensor *input,
    const Tensor *pre_output,
    const Tensor *weight,
    const Tensor *bias,
    const Tensor *pre_cell,
    Tensor *cell,
    Tensor *output,
    StatsFuture *future) {
  MACE_CHECK(pre_output->dim_size() == 2 && pre_output->dim(1) % 4 == 0,
             "LSTM hidden units should be a multiple of 4");

  const index_t height = input->dim(0);
  const index_t width = input->dim(1);
  const index_t hidden_units = pre_output->dim(1);
  const index_t w_blocks = hidden_units >> 2;

  auto runtime = context_->device()->opencl_runtime();

  if (kernel_.get() == nullptr) {
    std::set<std::string> built_options;
    OUT_OF_RANGE_CONFIG(kernel_error_, context_);
    NON_UNIFORM_WG_CONFIG;
    auto dt = DataTypeToEnum<T>::value;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("lstmcell");
    built_options.emplace("-Dlstmcell=" + kernel_name);
    built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    built_options.emplace("-DCMD_DATA_TYPE=" + DtToUpCompatibleCLCMDDt(dt));

    MACE_RETURN_IF_ERROR(runtime->BuildKernel("lstmcell", kernel_name,
                                              built_options, &kernel_));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(kernel_));
  }

  const uint32_t gws[2] = {static_cast<uint32_t>(w_blocks),
                           static_cast<uint32_t>(height)};

  if (!IsVecEqual(input_shape_, input->shape())) {
    std::vector<index_t> output_shape_padded = {height, 1, 1, hidden_units};
    std::vector<size_t> output_image_shape;
    CalImage2DShape(output_shape_padded, BufferType::IN_OUT_CHANNEL,
                    &output_image_shape);
    MACE_RETURN_IF_ERROR(output->ResizeImage(pre_output->shape(),
                                             output_image_shape));
    MACE_RETURN_IF_ERROR(cell->ResizeImage(pre_cell->shape(),
                                           output_image_shape));

    uint32_t idx = 0;
    OUT_OF_RANGE_SET_ARG;
    SET_2D_GWS_ARGS(kernel_);
    kernel_.setArg(idx++, *(input->opencl_image()));
    kernel_.setArg(idx++, *(pre_output->opencl_image()));
    kernel_.setArg(idx++, *(weight->opencl_image()));
    kernel_.setArg(idx++, *(bias->opencl_image()));
    kernel_.setArg(idx++, *(pre_cell->opencl_image()));
    kernel_.setArg(idx++, static_cast<float>(forget_bias_));
    kernel_.setArg(idx++, static_cast<int32_t>(width));
    kernel_.setArg(idx++, static_cast<int32_t>(hidden_units));
    kernel_.setArg(idx++, static_cast<int32_t>(RoundUpDiv4(width)));
    kernel_.setArg(idx++, *(cell->opencl_image()));
    kernel_.setArg(idx++, *(output->opencl_image()));

    input_shape_ = input->shape();
  }

  const std::vector<uint32_t> lws = {kwg_size_ / 16, 16, 0};
  std::string tuning_key =
      Concat("lstmcell_opencl_kernel", output->dim(0), output->dim(1));
  MACE_RETURN_IF_ERROR(TuningOrRun2DKernel(runtime, kernel_, tuning_key,
                                           gws, lws, future));
  OUT_OF_RANGE_VALIDATION(kernel_error_);

  return MACE_SUCCESS;
}

template struct LSTMCellFunctor<DeviceType::GPU, float>;

template struct LSTMCellFunctor<DeviceType::GPU, half>;

}  // namespace kernels
}  // namespace mace
