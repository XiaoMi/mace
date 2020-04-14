// Copyright 2019 The MACE Authors. All Rights Reserved.
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

template<DeviceType D, typename T>
class SelectOp;

template<class T>
class SelectOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit SelectOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    if (this->InputSize() == 1) {
      return RunWithNoData(context);
    } else {
      return RunWithData(context);
    }
  }

  MaceStatus RunWithNoData(OpContext *context) {
    const Tensor *condition = this->Input(CONDITION);
    Tensor *output = this->Output(OUTPUT);
    const index_t condition_rank = condition->dim_size();
    MACE_RETURN_IF_ERROR(output->Resize({condition->size(), condition_rank}));
    T *output_data = output->mutable_data<T>();
    const bool *condition_data = condition->data<bool>();

    index_t i = 0;
    if (condition_rank == 1) {
      const index_t channel = condition->dim(0);
      for (index_t c = 0; c < channel; ++c) {
        if (condition_data[c]) {
          output_data[i++] = c;
        }
      }
    } else if (condition_rank == 2) {
      const index_t width = condition->dim(0);
      const index_t channel = condition->dim(1);
      for (index_t w = 0; w < width; ++w) {
        index_t w_base = w * channel;
        for (index_t c = 0; c < channel; ++c) {
          if (condition_data[w_base + c]) {
            output_data[i++] = w;
            output_data[i++] = c;
          }
        }
      }
    } else if (condition_rank == 3) {
      const index_t height = condition->dim(0);
      const index_t width = condition->dim(1);
      const index_t channel = condition->dim(2);
      for (index_t h = 0; h < height; ++h) {
        index_t h_base = h * width;
        for (index_t w = 0; w < width; ++w) {
          index_t w_base = (w + h_base) * channel;
          for (index_t c = 0; c < channel; ++c) {
            if (condition_data[w_base + c]) {
              output_data[i++] = h;
              output_data[i++] = w;
              output_data[i++] = c;
            }
          }
        }
      }
    } else if (condition_rank == 4) {
      const index_t batch = condition->dim(0);
      const index_t height = condition->dim(1);
      const index_t width = condition->dim(2);
      const index_t channel = condition->dim(3);
      for (index_t b = 0; b < batch; ++b) {
        index_t b_base = b * height;
        for (index_t h = 0; h < height; ++h) {
          index_t h_base = (b_base + h) * width;
          for (index_t w = 0; w < width; ++w) {
            index_t w_base = (w + h_base) * channel;
            for (index_t c = 0; c < channel; ++c) {
              if (condition_data[w_base + c]) {
                output_data[i++] = b;
                output_data[i++] = h;
                output_data[i++] = w;
                output_data[i++] = c;
              }
            }
          }
        }
      }
    } else {
      const index_t condition_size = condition->size();
      const index_t condition_rank = condition->dim_size();
      auto div_buffer = context->device()->scratch_buffer();
      div_buffer->Rewind();
      MACE_RETURN_IF_ERROR(div_buffer->GrowSize(
          condition_rank * sizeof(index_t)));
      index_t *div_ptr = div_buffer->mutable_data<index_t>();
      div_ptr[condition_rank - 1] = 1;
      for (index_t dim = condition_rank - 1; dim > 0; --dim) {
        div_ptr[dim - 1] = div_ptr[dim] * condition->dim(dim);
      }
      for (index_t c = 0; c < condition_size; ++c) {
        if (condition_data[c]) {
          auto remainder = c;
          for (index_t dim = 0; dim < condition_rank; ++dim) {
            output_data[i++] = remainder / div_ptr[dim];
            remainder = remainder % div_ptr[dim];
          }
        }
      }
    }

    MACE_RETURN_IF_ERROR(output->Resize({i / condition_rank, condition_rank}));
    return MaceStatus::MACE_SUCCESS;
  }

  bool CheckDataValid(const Tensor *condition,
                      const Tensor *x, const Tensor *y) {
    const index_t x_rank = x->dim_size();
    const index_t y_rank = y->dim_size();
    const index_t condition_rank = condition->dim_size();
    MACE_CHECK(condition_rank <= x_rank && x_rank == y_rank);

    for (index_t i = 0; i < condition_rank; ++i) {
      MACE_CHECK(condition->dim(i) == x->dim(i),
                 "dimensions are not equal: ",
                 MakeString(condition->shape()),
                 " vs. ",
                 MakeString(x->shape()));
    }

    for (index_t i = 0; i < x_rank; ++i) {
      MACE_CHECK(y->dim(i) == x->dim(i), "dimensions are not equal: ",
                 MakeString(y->shape()), " vs. ", MakeString(x->shape()));
    }

    return true;
  }

  MaceStatus RunWithData(OpContext *context) {
    const Tensor *condition = this->Input(CONDITION);
    const Tensor *x = this->Input(X);
    const Tensor *y = this->Input(Y);
    MACE_ASSERT(CheckDataValid(condition, x, y));

    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->Resize(x->shape()));
    T *output_data = output->mutable_data<T>();
    const bool *condition_data = condition->data<bool>();
    const T *x_data = x->data<T>();
    const T *y_data = y->data<T>();

    const index_t condition_size = condition->size();
    const index_t x_size = x->size();
    utils::ThreadPool
        &thread_pool = context->device()->cpu_runtime()->thread_pool();
    if (condition_size == x_size) {
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          // LOG(INFO) << "condition_data[" << k << "] = " << condition_data[k];
          output_data[k] = condition_data[k] ? x_data[k] : y_data[k];
        }
      }, 0, x_size, 1);
    } else if (x_size > condition_size) {  // broadcast
      const auto block_size = x_size / condition_size;
      MACE_ASSERT(
          block_size > 1 && x_size % condition_size == 0,
          "x_size should be a multiple of condition_size and greater than 1");
      const auto raw_block_size = block_size * sizeof(T);
      thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
        for (index_t k = start; k < end; k += step) {
          auto offset = block_size * k;
          if (condition_data[k]) {
            memcpy(output_data + offset, x_data + offset, raw_block_size);
          } else {
            memcpy(output_data + offset, y_data + offset, raw_block_size);
          }
        }
      }, 0, condition_size, 1);
    } else {
      MACE_CHECK(false, "x_size should be bigger than condition_size");
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(CONDITION, X, Y);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterSelect(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "Select", SelectOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "Select", SelectOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
