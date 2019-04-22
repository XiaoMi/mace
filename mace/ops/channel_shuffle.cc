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

#include <memory>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/channel_shuffle.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ChannelShuffleOp;

template <typename T>
class ChannelShuffleOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ChannelShuffleOp(OpConstructContext *context)
      : Operation(context),
        groups_(Operation::GetOptionalArg<int>("group", 1)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim(1) % groups_ == 0,
               "input channels must be an integral multiple of group. ",
               input->dim(1));
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard logits_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    index_t batch = input->dim(0);
    index_t channels = input->dim(1);
    index_t height = input->dim(2);
    index_t width = input->dim(3);

    index_t image_size = height * width;
    index_t batch_size = channels * image_size;
    index_t channels_per_group = channels / groups_;

    for (index_t b = 0; b < batch; ++b) {
      for (index_t c = 0; c < channels; ++c) {
        index_t g = c % groups_;
        index_t idx = c / groups_;
        const T *in_ptr = input_ptr + b * batch_size
            + (g * channels_per_group + idx) * image_size;
        T *out_ptr = output_ptr + b * batch_size + c * image_size;
        memcpy(out_ptr, in_ptr, image_size * sizeof(float));
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  const int groups_;
};


#ifdef MACE_ENABLE_OPENCL
template <typename T>
class ChannelShuffleOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit ChannelShuffleOp(OpConstructContext *context)
      : Operation(context) {
    const int groups = Operation::GetOptionalArg<int>("group", 1);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ChannelShuffleKernel<T>>(groups);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLChannelShuffleKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterChannelShuffle(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ChannelShuffle",
                   ChannelShuffleOp, DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "ChannelShuffle",
                   ChannelShuffleOp, DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "ChannelShuffle",
                   ChannelShuffleOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL

  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("ChannelShuffle")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return { DeviceType::CPU, DeviceType::GPU };
                }
                int groups = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                    *op, "group", 1);
                if (op->output_shape(0).dims_size() != 4) {
                  return { DeviceType::CPU };
                }
                index_t channels = op->output_shape(0).dims(3);
                index_t channels_per_group = channels / groups;
                if (groups % 4 != 0 || channels_per_group % 4 != 0) {
                  return { DeviceType::CPU };
                }
                return { DeviceType::CPU, DeviceType::GPU };
              }));
}

}  // namespace ops
}  // namespace mace
