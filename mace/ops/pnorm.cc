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

// This Op is for PNormComponent in Kaldi.
// The input-dim must be dividable by output-dim.
// The output will be divided to output-dim group,
// so input-dim should be dividable by output-dim.
// For each row:
// p is 0: output[i] = sum(abs(input[i*group + j]) > 0)
// p is 1: output[i] = sum(abs(input[i*group + j]))
// p is 2: output[i] = sqrt(sum(input[i * group + j] * input[i * group + j])),
// for j = (0 : group - 1)
// p's default value is 2.

#include <functional>
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class PNormOp;

template<typename T>
class PNormOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit PNormOp(OpConstructContext *context)
      : Operation(context),
        p_(Operation::GetOptionalArg<int>("p", 2)),
        output_dim_(Operation::GetOptionalArg<int>("output_dim", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    const std::vector<index_t> &input_shape = input->shape();
    const index_t dim_size = input_shape.size();
    MACE_CHECK(dim_size >= 1, "PNorm only supports input dim size >= 1");
    std::vector<index_t> output_shape(input_shape);
    const index_t input_dim = input_shape[dim_size - 1];
    MACE_CHECK(output_dim_ > 0,
               "Output dim should be greater than zero.");
    MACE_CHECK(input_dim % output_dim_ == 0 && output_dim_ < input_dim,
               "PNorm's input dim should be a multiple of output dim.");
    const index_t group_size = input_dim / output_dim_;
    output_shape[dim_size - 1] = output_dim_;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_output(output);

    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();
    const index_t bh =
        std::accumulate(input->shape().begin(), input->shape().end() - 1, 1,
                        std::multiplies<index_t>());

    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();

    if (p_ == 0) {
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            const T *in_base = input_data + i * input_dim + j * group_size;
            T *out_base = output_data + i * output_dim_;
            T temp_result = 0;
            for (index_t g = 0; g < group_size; ++g) {
              T value =
                  (std::fabs(in_base[g])
                      > std::numeric_limits<float>::epsilon()) ? 1.0f : 0.0f;
              temp_result += value;
            }
            out_base[j] = temp_result;
          }
        }
      }, 0, bh, 1, 0, output_dim_, 1);

    } else if (p_ == 1) {
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            const T *in_base = input_data + i * input_dim + j * group_size;
            T *out_base = output_data + i * output_dim_;
            T temp_result = 0;
            for (index_t g = 0; g < group_size; ++g) {
              temp_result += std::abs(in_base[g]);;
            }
            out_base[j] = temp_result;
          }
        }
      }, 0, bh, 1, 0, output_dim_, 1);
    } else if (p_ == 2) {
      thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                                index_t start1, index_t end1, index_t step1) {
        for (index_t i = start0; i < end0; i += step0) {
          for (index_t j = start1; j < end1; j += step1) {
            const T *in_base = input_data + i * input_dim + j * group_size;
            T *out_base = output_data + i * output_dim_;
            T temp_result = 0;
            for (index_t g = 0; g < group_size; ++g) {
              temp_result += in_base[g] * in_base[g];
            }
            out_base[j] = std::sqrt(temp_result);
          }
        }
      }, 0, bh, 1, 0, output_dim_, 1);
    } else {
      LOG(FATAL) << "PNorm's p should be 0, 1 or 2, here p is: " << p_;
    }
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int p_;
  int output_dim_;
};

void RegisterPNorm(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "PNorm", PNormOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
