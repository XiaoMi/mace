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

#include <functional>
#include <memory>
#include <vector>

#include "mace/core/operator.h"
#include "mace/ops/activation.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/bias_add.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

template <DeviceType D, class T>
class BiasAddOp;

template <>
class BiasAddOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit BiasAddOp(OpConstructContext *context)
      : Operation(context),
        data_format_(static_cast<DataFormat>(Operation::GetOptionalArg<int>(
                     "data_format", NHWC))) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    const Tensor *bias = this->Input(1);

    MACE_CHECK(bias->dim_size() == 1, "bias must be 1-dimensional. ",
               bias->dim_size());

    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard bias_mapper(bias);
    Tensor::MappingGuard output_mapper(output);

    const float *input_ptr = input->data<float>();
    const float *bias_ptr = bias->data<float>();
    float *output_ptr = output->mutable_data<float>();

    if (input->dim_size() == 4 && data_format_ == NCHW) {
      const index_t batch = input->dim(0);
      const index_t channels = input->dim(1);
      const index_t height_width = input->dim(2) * input->dim(3);

#pragma omp parallel for collapse(2)
      for (index_t n = 0; n < batch; ++n) {
        for (index_t c = 0; c < channels; ++c) {
          for (index_t hw = 0; hw < height_width; ++hw) {
            index_t pos = (n * channels + c) * height_width + hw;
            output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
          }
        }
      }
    } else {
      const std::vector<index_t> &shape = input->shape();
      const index_t fused_batch = std::accumulate(
          shape.begin(), shape.end() - 1, 1, std::multiplies<index_t>());
      const index_t channels = *shape.rbegin();
#pragma omp parallel for
      for (index_t n = 0; n < fused_batch; ++n) {
        index_t pos = n * channels;
        for (index_t c = 0; c < channels; ++c) {
          output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
          ++pos;
        }
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  DataFormat data_format_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class BiasAddOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit BiasAddOp(OpConstructContext *context)
      : Operation(context),
        data_format_(static_cast<DataFormat>(Operation::GetOptionalArg<int>(
            "data_format", NHWC))) {
    MemoryType mem_type;
    if (context->device()->gpu_runtime()->UseImageMemory()) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_.reset(new opencl::image::BiasAddKernel<T>);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(TransformFilter<T>(
        context, operator_def_.get(), 1, OpenCLBufferType::ARGUMENT, mem_type)
                   == MaceStatus::MACE_SUCCESS);
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *bias = this->Input(1);

    MACE_CHECK(bias->dim_size() == 1, "bias must be 1-dimensional. ",
               bias->dim_size());

    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    MACE_CHECK(input->dim_size() == 4 && data_format_ == NHWC,
               "gpu only support biasadd for 4-dimensional NHWC format tensor");
    return kernel_->Compute(context, input, bias, output);
  }

 private:
  DataFormat data_format_;
  std::unique_ptr<OpenCLBiasAddKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterBiasAdd(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "BiasAdd", BiasAddOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "BiasAdd", BiasAddOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "BiasAdd", BiasAddOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
