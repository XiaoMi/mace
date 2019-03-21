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

// This Op is for IfDefined descriptor in Kaldi.
// It defines time offset.
// If time index <= offset, using zeros as output.

#include <functional>
#include <memory>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class DelayOp;

template <typename T>
class DelayOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit DelayOp(OpConstructContext *context)
      : Operation(context),
        offset_(Operation::GetOptionalArg<int>("offset", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(offset_ < 0, "offset param should be negative.");

    index_t rank = input->dim_size();
    MACE_CHECK(rank >= 2, "input's rank should >= 2.");
    const std::vector<index_t> &input_shape = input->shape();
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t chunk = input_shape[rank - 2];
    const index_t dim = input_shape[rank - 1];
    MACE_RETURN_IF_ERROR(output->ResizeLike(input));
    output->Clear();

    if (chunk <= -offset_)
      return MaceStatus::MACE_SUCCESS;

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                              index_t start1, index_t end1, index_t step1) {
      for (index_t i = start0; i < end0; i += step0) {
        for (index_t j = start1; j < end1; j += step1) {
          memcpy(output_data + (i * chunk + j - offset_) * dim,
                 input_data + (i * chunk + j) * dim,
                 dim * sizeof(T));
        }
      }
    }, 0, batch, 1, 0, chunk + offset_, 1);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int offset_;
};

void RegisterDelay(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Delay", DelayOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
