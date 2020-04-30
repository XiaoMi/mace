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

#include "mace/ops/opencl/buffer/pooling.h"

#include "mace/runtimes/opencl/opencl_runtime.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

MaceStatus PoolingKernel::Compute(
    OpContext *context,
    const Tensor *input,
    const PoolingType pooling_type,
    const int *kernels,
    const int *strides,
    const Padding &padding_type,
    const std::vector<int> &padding_data,
    const int *dilations,
    const RoundType round_type,
    Tensor *output) {
  MACE_CHECK(dilations[0] == 1 && dilations[1] == 1)
    << "Pooling opencl kernel not support dilation yet";

  StatsFuture pad_future, pooling_future;

  index_t input_channels = input->dim(3);

  std::vector<index_t> output_shape(4);
  std::vector<index_t> filter_shape = {input->dim(3), input->dim(3),
                                       kernels[0], kernels[1]};

  std::vector<int> paddings(2);
  if (padding_data.empty()) {
    ops::CalcNHWCPaddingAndOutputSize(
        input->shape().data(), filter_shape.data(), dilations, strides,
        padding_type, output_shape.data(), paddings.data());
  } else {
    paddings = padding_data;
    CalcOutputSize(input->shape().data(), filter_shape.data(),
                   padding_data.data(), dilations, strides, round_type,
                   output_shape.data());
  }

  MACE_RETURN_IF_ERROR(output->Resize(output_shape));

  // Mark whether input changed or not
  bool input_changed = IsResetArgsNeeded(context, input_shape_, input->shape());
  input_shape_ = input->shape();
  auto executor = OpenclRuntime::Get(context)->GetOpenclExecutor();

  // pad input
  std::vector<index_t> padded_input_shape = input->shape();
  padded_input_shape[3] = RoundUp<index_t>(input_channels, 4);

  const Tensor *padded_input_ptr = input;
  // pad input
  std::unique_ptr<Tensor> padded_input;
  if (padded_input_shape[3] != input_channels) {
    index_t padded_input_size =
        std::accumulate(padded_input_shape.begin(),
                        padded_input_shape.end(),
                        1, std::multiplies<index_t>()) +
            MACE_EXTRA_BUFFER_PAD_SIZE / GetEnumTypeSize(input->dtype());

    // Init scratch buffer
    auto *runtime = context->runtime();
    padded_input.reset(new Tensor(
        runtime, input->dtype(), output->memory_type(), {padded_input_size}));
    runtime->AllocateBufferForTensor(padded_input.get(), RENT_SCRATCH);

    padded_input->Resize(padded_input_shape);
    PadInput(context, &kernels_[0], input, 0, 0,
             input_changed, padded_input.get(), &pad_future);
    padded_input_ptr = padded_input.get();
  }

  cl::Kernel *kernel = &kernels_[1];
  MACE_OUT_OF_RANGE_DEFINITION

  if (kernel->get() == nullptr) {
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pooling");
    built_options.emplace("-Dpooling=" + kernel_name);
    auto input_dtype = input->dtype();
    auto input_dt = DtToCLDt(input_dtype);
    built_options.emplace("-DIN_DATA_TYPE=" + input_dt);
    auto output_dtype = output->dtype();
    built_options.emplace("-DOUT_DATA_TYPE=" + DtToCLDt(output_dtype));
    if (pooling_type == MAX && input_dtype == output_dtype) {
      built_options.emplace("-DDATA_TYPE=" + input_dt);
    } else {
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(DT_FLOAT));
    }
    if (pooling_type == AVG) {
      built_options.emplace("-DPOOL_AVG");
    }
    MACE_RETURN_IF_ERROR(executor->BuildKernel("pooling_buffer",
                                               kernel_name,
                                               built_options,
                                               kernel));

    kwg_size_ =
        static_cast<uint32_t>(executor->GetKernelMaxWorkGroupSize(*kernel));
  }

  const uint32_t gws[3] = {
      static_cast<uint32_t>(RoundUpDiv4(output->dim(3))),
      static_cast<uint32_t>(output->dim(2)),
      static_cast<uint32_t>(output->dim(0) * output->dim(1)),
  };

  MACE_OUT_OF_RANGE_INIT(*kernel);
  if (input_changed) {
    uint32_t idx = 0;
    MACE_BUFF_OUT_OF_RANGE_SET_ARGS(*kernel, output->size());
    MACE_SET_3D_GWS_ARGS(*kernel, gws);
    kernel->setArg(idx++, *(padded_input_ptr->memory<cl::Buffer>()));
    kernel->setArg(idx++, static_cast<int32_t>(padded_input_ptr->dim(1)));
    kernel->setArg(idx++, static_cast<int32_t>(padded_input_ptr->dim(2)));
    kernel->setArg(idx++, static_cast<int32_t>(padded_input_ptr->dim(3)));
    kernel->setArg(idx++, static_cast<int32_t>(output->dim(1)));
    kernel->setArg(idx++, static_cast<int32_t>(output->dim(3)));
    kernel->setArg(idx++, paddings[0] / 2);
    kernel->setArg(idx++, paddings[1] / 2);
    kernel->setArg(idx++, strides[0]);
    kernel->setArg(idx++, strides[1]);
    kernel->setArg(idx++, kernels[0]);
    kernel->setArg(idx++, kernels[1]);
    kernel->setArg(idx++, *(output->memory<cl::Buffer>()));
  }

  const std::vector<uint32_t> lws = {4, 4, 4, 0};
  std::string tuning_key =
      Concat("pooling_opencl_kernel_", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(executor, *kernel, tuning_key,
                                           gws, lws, &pooling_future));
  MACE_OUT_OF_RANGE_VALIDATION;
  MergeMultipleFutureWaitFn({pad_future, pooling_future}, context->future());

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace
