// Copyright 2022 The MACE Authors. All Rights Reserved.
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

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/core/tensor.h"

namespace mace {
namespace ops {

template<RuntimeType D, typename T>
class WhereOp;

template<class T>
class WhereOp<RuntimeType::RT_CPU, T> : public Operation {
 public:
  explicit WhereOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_CHECK(this->InputSize() == 3);
    const Tensor *condition = this->Input(CONDITION);
    const Tensor *x = this->Input(X);
    const Tensor *y = this->Input(Y);
    Tensor *output = this->Output(OUTPUT);

    const bool *condition_data = condition->data<bool>();
    const T *x_data = x->data<T>();
    const T *y_data = y->data<T>();

    const index_t condition_size = condition->size();
    const index_t x_size = x->size();
    const index_t y_size = y->size();
    MACE_CHECK(condition_size <= y_size || condition_size <= x_size);
    MACE_CHECK(x_size == y_size || x_size == 1 || y_size == 1);
    if (x_size == 1) {
      MACE_RETURN_IF_ERROR(output->Resize(y->shape()));
    } else {
      MACE_RETURN_IF_ERROR(output->Resize(x->shape()));
    }
    T *output_data = output->mutable_data<T>();
    utils::ThreadPool &thread_pool = context->runtime()->thread_pool();
    if (x_size == 1 && condition_size == y_size) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          output_data[k] = condition_data[k] ? x_data[0] : y_data[k];
        }
      }, 0, y_size, 1);
    } else if (y_size == 1 && condition_size == x_size) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          output_data[k] = condition_data[k] ? x_data[k] : y_data[0];
        }
      }, 0, x_size, 1);
    } else if (y_size == x_size && condition_size == x_size) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          output_data[k] = condition_data[k] ? x_data[k] : y_data[k];
        }
      }, 0, y_size, 1);
    } else if (x_size == y_size && y_size > condition_size) {  // broadcast
      const auto block_size = y_size / condition_size;
      MACE_ASSERT(
          block_size > 1 && y_size % condition_size == 0,
          "y_size should be a multiple of condition_size and greater than 1");
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          if (condition_data[k]) {
            for (uint32_t l = 0; l < block_size; l += 1) {
              output_data[l*condition_size + k] = x_data[l*condition_size + k];
            }
          } else {
            for (uint32_t l = 0; l < block_size; l += 1) {
              output_data[l*condition_size + k] = y_data[l*condition_size + k];
            }
          }
        }
      }, 0, condition_size, 1);
    } else if (x_size == 1 && y_size > condition_size) {  // broadcast
      const auto block_size = y_size / condition_size;
      MACE_ASSERT(
          block_size > 1 && y_size % condition_size == 0,
          "y_size should be a multiple of condition_size and greater than 1");
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          if (condition_data[k]) {
            for (uint32_t l = 0; l < block_size; l += 1) {
              output_data[l*condition_size + k] = x_data[0];
            }
          } else {
            for (uint32_t l = 0; l < block_size; l += 1) {
              output_data[l*condition_size + k] = y_data[l*condition_size + k];
            }
          }
        }
      }, 0, condition_size, 1);
    } else if (y_size == 1 && x_size > condition_size) {  // broadcast
      const auto block_size = x_size / condition_size;
      MACE_ASSERT(
          block_size > 1 && x_size % condition_size == 0,
          "x_size should be a multiple of condition_size and greater than 1");
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          if (condition_data[k]) {
            for (uint32_t l = 0; l < block_size; l += 1) {
              output_data[l*condition_size + k] = x_data[l*condition_size + k];
            }
          } else {
            for (uint32_t l = 0; l < block_size; l += 1) {
              output_data[l*condition_size + k] = y_data[0];
            }
          }
        }
      }, 0, condition_size, 1);
    } else {
      MACE_CHECK(false, "Input shape is not support");
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(CONDITION, X, Y);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterWhere(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Where", WhereOp,
                   RuntimeType::RT_CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Where", WhereOp,
                        RuntimeType::RT_CPU);
}

}  // namespace ops
}  // namespace mace
