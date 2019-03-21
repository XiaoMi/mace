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

// This Op is for SumGroupComponent in Kaldi.
// It's used to sum up groups of posteriors,
// and to introduce a kind of Gaussian-mixture-model-like
// idea into neural nets.

#include <functional>
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class SumGroupOp;

template <typename T>
class SumGroupOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit SumGroupOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(this->InputSize() >= 2,
               "SumGroup should have at least 2 inputs.");
    const Tensor *input = this->Input(0);
    // Sizes-input gets a vector saying, for
    // each output-dim, how many
    // inputs data were summed over.
    const Tensor *sizes = this->Input(1);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() >= 1,
               "SumGroup's input's rank should be >= 1.");
    MACE_CHECK(sizes->dim_size() == 1,
               "SumGroup's sizes input should be a vector.");

    const std::vector<index_t> &input_shape = input->shape();
    const index_t bh =
        std::accumulate(input_shape.begin(), input_shape.end() - 1, 1,
                        std::multiplies<index_t>());
    std::vector<index_t> output_shape(input_shape);
    const index_t output_dim = sizes->dim(0);
    const index_t dim_size = input->dim_size();
    const index_t input_dim = input_shape[dim_size -1];
    output_shape[dim_size - 1] = output_dim;

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    Tensor::MappingGuard guard_input(input);
    Tensor::MappingGuard guard_sizes(sizes);
    Tensor::MappingGuard guard_output(output);
    const T *input_data = input->data<T>();
    const int *sizes_data = sizes->data<int>();
    T *output_data = output->mutable_data<T>();

    std::vector<std::pair<int, int>>
        sum_indexes(static_cast<size_t >(output_dim));

    int cur_index = 0;
    for (index_t i = 0; i < output_dim; ++i) {
      int size_value = sizes_data[i];
      MACE_CHECK(size_value > 0, "size value should be > 0");
      sum_indexes[i].first = cur_index;
      cur_index += size_value;
      sum_indexes[i].second = cur_index;
      MACE_CHECK(cur_index <= input_dim)
        << "size value over-ranged:" << cur_index << "<=" << input_dim;
    }
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        for (index_t j = start1; j < end1; j += step1) {
          int start_col = sum_indexes[j].first;
          int end_col = sum_indexes[j].second;
          T sum = 0;
          for (int src_col = start_col; src_col < end_col; ++src_col) {
            sum += input_data[i * input_dim + src_col];
          }
          output_data[i * output_dim + j] = sum;
        }
      }
    }, 0, bh, 1, 0, output_dim, 1);

    return MaceStatus::MACE_SUCCESS;
  }
};

void RegisterSumGroup(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "SumGroup", SumGroupOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
