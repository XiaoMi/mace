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
#ifndef MACE_OPS_OPENCL_BUFFER_POOLING_H_
#define MACE_OPS_OPENCL_BUFFER_POOLING_H_

#include "mace/ops/opencl/pooling.h"

#include <functional>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "mace/ops/opencl/buffer/utils.h"
#include "mace/ops/opencl/helper.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace opencl {
namespace buffer {

template <typename T>
class PoolingKernel : public OpenCLPoolingKernel {
 public:
  PoolingKernel() : old_scratch_size_(0) {}
  MaceStatus Compute(
      OpContext *context,
      const Tensor *input,
      const PoolingType pooling_type,
      const int *kernels,
      const int *strides,
      const Padding &padding_type,
      const std::vector<int> &padding_data,
      const int *dilations,
      const RoundType round_type,
      Tensor *output) override;

 private:
  index_t old_scratch_size_;
  cl::Kernel kernels_[2];
  uint32_t kwg_size_;
  std::vector<index_t> input_shape_;
};

template <typename T>
MaceStatus PoolingKernel<T>::Compute(
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
  bool input_changed = !IsVecEqual(input_shape_, input->shape());
  input_shape_ = input->shape();

  auto runtime = context->device()->gpu_runtime()->opencl_runtime();

  // pad input
  std::vector<index_t> padded_input_shape = input->shape();
  padded_input_shape[3] = RoundUp<index_t>(input_channels, 4);

  const Tensor *padded_input_ptr = input;
  // pad input
  std::unique_ptr<Tensor> padded_input;
  if (padded_input_shape[3] != input_channels) {
    index_t total_scratch_size = 0;
    index_t padded_input_size = 0;

    padded_input_size =
        std::accumulate(padded_input_shape.begin(),
                        padded_input_shape.end(),
                        1,
                        std::multiplies<index_t>())
            * GetEnumTypeSize(input->dtype()) + MACE_EXTRA_BUFFER_PAD_SIZE;
    total_scratch_size += padded_input_size;

    // Init scratch buffer
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(total_scratch_size);
    if (old_scratch_size_ != scratch->size()) {
      input_changed |= scratch->size() != old_scratch_size_;
      old_scratch_size_ = scratch->size();
    }

    padded_input = make_unique<Tensor>(scratch->Scratch(padded_input_size),
                                       input->dtype());

    padded_input->Resize(padded_input_shape);
    PadInput(context, &kernels_[0], input, 0, 0,
             input_changed, padded_input.get(), &pad_future);
    padded_input_ptr = padded_input.get();
  }

  cl::Kernel *kernel = &kernels_[1];
  MACE_OUT_OF_RANGE_DEFINITION

  if (kernel->get() == nullptr) {
    const DataType dt = DataTypeToEnum<T>::value;
    std::set<std::string> built_options;
    MACE_OUT_OF_RANGE_CONFIG;
    MACE_NON_UNIFORM_WG_CONFIG;
    std::string kernel_name = MACE_OBFUSCATE_SYMBOL("pooling");
    built_options.emplace("-Dpooling=" + kernel_name);

    if (pooling_type == MAX && input->dtype() == output->dtype()) {
      built_options.emplace("-DIN_DATA_TYPE=" +
          DtToCLDt(input->dtype()));
      built_options.emplace("-DOUT_DATA_TYPE=" + DtToCLDt(dt));
      built_options.emplace("-DDATA_TYPE=" + DtToCLDt(dt));
    } else {
      built_options.emplace("-DIN_DATA_TYPE=" +
          DtToCLDt(input->dtype()));
      built_options.emplace("-DOUT_DATA_TYPE=" + DtToCLDt(dt));
      built_options.emplace("-DDATA_TYPE=" + DtToUpCompatibleCLDt(dt));
    }
    if (pooling_type == AVG) {
      built_options.emplace("-DPOOL_AVG");
    }
    MACE_RETURN_IF_ERROR(runtime->BuildKernel("pooling_buffer",
                                              kernel_name,
                                              built_options,
                                              kernel));

    kwg_size_ =
        static_cast<uint32_t>(runtime->GetKernelMaxWorkGroupSize(*kernel));
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
    kernel->setArg(idx++, *(padded_input_ptr->opencl_buffer()));
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
    kernel->setArg(idx++, *(output->opencl_buffer()));
  }

  const std::vector<uint32_t> lws = {4, 4, 4, 0};
  std::string tuning_key =
      Concat("pooling_opencl_kernel_", output->dim(0), output->dim(1),
             output->dim(2), output->dim(3));
  MACE_RETURN_IF_ERROR(TuningOrRun3DKernel(runtime, *kernel, tuning_key,
                                           gws, lws, &pooling_future));
  MACE_OUT_OF_RANGE_VALIDATION
  MergeMultipleFutureWaitFn({pad_future, pooling_future}, context->future());

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace buffer
}  // namespace opencl
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_OPENCL_BUFFER_POOLING_H_
