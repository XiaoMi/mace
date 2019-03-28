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


#include "mace/core/operator.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class InferConv2dShapeOp : public Operation {
 public:
  explicit InferConv2dShapeOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4);
    output->Resize({input->dim_size()});
    Tensor::MappingGuard output_guard(output);
    int32_t *output_data = output->mutable_data<int32_t>();

    auto has_data_format =
        Operation::GetOptionalArg<int>("has_data_format", 0);
    const bool isNCHW = (has_data_format &&
        input->data_format() == DataFormat::NCHW);

    Padding padding_type =
        static_cast<Padding>(Operation::GetOptionalArg<int>(
            "padding", static_cast<int>(SAME)));
    const std::vector<int32_t> paddings =
        Operation::GetRepeatedArgs<int32_t>("padding_values");
    const std::vector<int32_t> kernels =
        Operation::GetRepeatedArgs<int32_t>("kernels");
    const std::vector<int32_t> strides =
        Operation::GetRepeatedArgs<int32_t>("strides", {1, 1});
    const int32_t out_batch = static_cast<int32_t>(input->dim(0));
    const int32_t out_channel = static_cast<int32_t>(kernels[0]);

    int32_t in_h = 0, in_w = 0, in_c = 0;
    if (isNCHW) {  // NCHW
      in_c = static_cast<int32_t>(input->dim(1));
      in_h = static_cast<int32_t>(input->dim(2));
      in_w = static_cast<int32_t>(input->dim(3));
    } else {
      in_h = static_cast<int32_t>(input->dim(1));
      in_w = static_cast<int32_t>(input->dim(2));
      in_c = static_cast<int32_t>(input->dim(3));
    }
    MACE_CHECK(in_c == kernels[1],
               "different number of input channels between input and kernel");
    int32_t out_h = 0, out_w = 0;
    if (!paddings.empty()) {
      out_h = (in_h - kernels[2] + paddings[0]) / strides[0] + 1;
      out_w = (in_w - kernels[3]  + paddings[1]) / strides[1] + 1;
    } else {
      switch (padding_type) {
        case SAME:
          out_h = (in_h + strides[0] - 1) / strides[0];
          out_w = (in_w + strides[1] - 1) / strides[1];
          break;
        case VALID:
          out_h = (in_h - kernels[2] + 1) / strides[0];
          out_w = (in_w - kernels[3] + 1) / strides[1];
          break;
        default:
          MACE_NOT_IMPLEMENTED;
          break;
      }
    }

    if (isNCHW) {
      output_data[0] = out_batch;
      output_data[1] = out_channel;
      output_data[2] = out_h;
      output_data[3] = out_w;
    } else {
      output_data[0] = out_batch;
      output_data[1] = out_h;
      output_data[2] = out_w;
      output_data[3] = out_channel;
    }

    return MaceStatus::MACE_SUCCESS;
  }
};

void RegisterInferConv2dShape(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "InferConv2dShape",
                   InferConv2dShapeOp, DeviceType::CPU, float);
  MACE_REGISTER_OP(op_registry, "InferConv2dShape",
                   InferConv2dShapeOp, DeviceType::CPU, int32_t);
#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "InferConv2dShape",
                   InferConv2dShapeOp, DeviceType::GPU, float);
  MACE_REGISTER_OP(op_registry, "InferConv2dShape",
                   InferConv2dShapeOp, DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
