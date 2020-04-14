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

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/activation.h"
#include "mace/ops/delegator/bias_add.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/buffer_transformer.h"
#include "mace/ops/opencl/image/bias_add.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

template<DeviceType D, class T>
class BiasAddOp;

template<class T>
class BiasAddOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit BiasAddOp(OpConstructContext *context)
      : Operation(context),
        has_data_format_(Operation::GetOptionalArg<int>("has_data_format", 0)),
        bias_add_delegator_(delegator::BiasAdd::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(BiasAdd, DeviceType::CPU, T, kCpuImplType),
            DelegatorParam())) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    const Tensor *bias = this->Input(1);
    Tensor *output = this->Output(0);

    if (input->dim_size() == 4 && (has_data_format_
        || input->data_format() == DataFormat::NCHW)) {  // NCHW
      MACE_CHECK(bias->dim_size() == 1 || bias->dim_size() == 2,
                 "bias must be 1-dimensional or n*c for caffee.",
                 MakeString(bias->shape()));
      bias_add_delegator_->Compute(context, input, bias, output);
    } else {  // NHWC
      MACE_CHECK(bias->dim_size() == 1 || bias->dim_size() == 2,
                 "bias must be 1 or 2 dimensionals for caffee.",
                 bias->dim_size(), MakeString(bias->shape()));
      // TODO(liyin): remove it and tranform bias to add (eltwise)
      MACE_RETURN_IF_ERROR(output->ResizeLike(input));

      Tensor::MappingGuard input_mapper(input);
      Tensor::MappingGuard bias_mapper(bias);
      Tensor::MappingGuard output_mapper(output);

      const T *input_ptr = input->data<T>();
      const T *bias_ptr = bias->data<T>();
      T *output_ptr = output->mutable_data<T>();

      const std::vector<index_t> &shape = input->shape();
      const index_t channels = *shape.rbegin();
      utils::ThreadPool
          &thread_pool = context->device()->cpu_runtime()->thread_pool();
      if (bias->dim_size() == 1) {
        const index_t fused_batch = std::accumulate(
            shape.begin(), shape.end() - 1, 1, std::multiplies<index_t>());
        thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
          for (index_t n = start; n < end; n += step) {
            index_t pos = n * channels;
            for (index_t c = 0; c < channels; ++c) {
              output_ptr[pos] = input_ptr[pos] + bias_ptr[c];
              ++pos;
            }
          }
        }, 0, fused_batch, 1);
      } else {  // bias is 2d
        const auto n = shape[0];
        MACE_CHECK(n == bias->shape()[0]);
        const index_t fused_hw = std::accumulate(
            shape.begin() + 1, shape.end() - 1, 1, std::multiplies<index_t>());
        const auto ch_size = bias->shape()[1];
        thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                  index_t start1, index_t end1, index_t step1) {
          for (index_t i = start0; i < end0; i += step0) {
            auto offset = i * fused_hw;
            auto bias_offset = i * ch_size;
            for (index_t j = start1; j < end1; j += step1) {
              index_t pos = (offset + i) * channels;
              for (index_t c = 0; c < channels; ++c, ++pos) {
                output_ptr[pos] = input_ptr[pos] + bias_ptr[bias_offset + c];
              }
            }
          }
        }, 0, n, 1, 0, fused_hw, 1);
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int has_data_format_;
  std::unique_ptr<delegator::BiasAdd> bias_add_delegator_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class BiasAddOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit BiasAddOp(OpConstructContext *context)
      : Operation(context),
        has_data_format_(Operation::GetOptionalArg<int>("has_data_format", 1)) {
    MemoryType mem_type = MemoryType::CPU_BUFFER;
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      mem_type = MemoryType::GPU_IMAGE;
      kernel_ = make_unique<opencl::image::BiasAddKernel>();
    } else {
      MACE_NOT_IMPLEMENTED;
    }

    // for const bias tensor
    if (context->workspace()->GetTensor(operator_def_->input(1)) != nullptr) {
      MACE_CHECK(TransformFilter(
          context, operator_def_.get(), 1, OpenCLBufferType::ARGUMENT, mem_type)
                     == MaceStatus::MACE_SUCCESS, "TransformFilter failed");
    }
  }

  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    const Tensor *bias = this->Input(1);
    Tensor *output = this->Output(0);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    MACE_CHECK(input->dim_size() == 4 && has_data_format_,
               "gpu only support biasadd for 4-dimensional NHWC format tensor");
    MACE_CHECK(bias->dim_size() == 1 || bias->dim_size() == 2,
               "bias must be 1-dimensional or 2-dimensional for caffee. ",
               MakeString(bias->shape()));
    return kernel_->Compute(context, input, bias, output);
  }

 private:
  int has_data_format_;
  std::unique_ptr<OpenCLBiasAddKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterBiasAdd(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "BiasAdd", BiasAddOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "BiasAdd", BiasAddOp, DeviceType::CPU);
  MACE_REGISTER_GPU_OP(op_registry, "BiasAdd", BiasAddOp);
  MACE_REGISTER_OP_CONDITION(
      op_registry,
      OpConditionBuilder("BiasAdd")
          .SetDevicePlacerFunc(
              [](OpConditionContext *context) -> std::set<DeviceType> {
                auto op = context->operator_def();
                if (op->output_shape_size() != op->output_size()) {
                  return {DeviceType::CPU, DeviceType::GPU};
                }
                int has_data_format =
                    ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                        *op, "has_data_format", 0);
                if (!has_data_format ||
                    op->output_shape(0).dims_size() != 4) {
                  LOG(INFO) << "BiasAdd only support cpu, has_data_format="
                            << has_data_format
                            << ", op->output_shape(0).dims_size()="
                            << op->output_shape(0).dims_size();
                  return {DeviceType::CPU};
                }
                return {DeviceType::CPU, DeviceType::GPU};
              }));
}

}  // namespace ops
}  // namespace mace
