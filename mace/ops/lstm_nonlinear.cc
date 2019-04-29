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

// This Op is for LstmNonlinearComponent in Kaldi.
// http://kaldi-asr.org/doc/nnet-simple-component_8h_source.html#l02164

#include <functional>
#include <memory>

#include "mace/core/operator.h"
#include "mace/ops/common/lstm.h"

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class LSTMNonlinearOp;

template<typename T>
class LSTMNonlinearOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit LSTMNonlinearOp(OpConstructContext *context)
      : Operation(context) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(INPUT);
    MACE_CHECK(this->InputSize() >= 2,
               "LSTMNonlinear should have at least 2 inputs.");
    const Tensor *params = this->Input(PARAMS);
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(input->dim_size() >= 2)
      << "The input dim size should >= 2";
    MACE_CHECK(params->dim_size() == 2)
      << "The params dim size should be 2";

    const std::vector<index_t> &input_shape = input->shape();
    const std::vector<index_t> &params_shape = params->shape();

    const index_t num_rows =
        std::accumulate(input_shape.begin(), input_shape.end() - 1, 1,
                        std::multiplies<index_t>());
    index_t rank = input->dim_size();
    const index_t input_cols = input_shape[rank - 1];
    const index_t cell_dim = input_cols / 5;
    bool embed_scales = input_cols == cell_dim * 5 + 3;
    const index_t params_stride = params_shape[1];

    MACE_CHECK(input_cols == (cell_dim * 5) || embed_scales);
    MACE_CHECK(params_shape[0] == 3 && params_shape[1] == cell_dim);

    const index_t output_dim = cell_dim * 2;
    std::vector<index_t> output_shape = input->shape();
    output_shape[rank - 1] = output_dim;
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard params_guard(params);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<T>();
    const float *params_data = params->data<T>();
    float *output_data = output->mutable_data<T>();

    for (int r = 0; r < num_rows; ++r) {
      const float *input_row = input_data + r * input_cols;
      const float *prev_row = input_row + 4 * cell_dim;
      const float *scale_data =
          embed_scales ? prev_row + cell_dim : nullptr;
      float *output_cell = output_data + r * output_dim;
      float *output_row = output_cell + cell_dim;
      LSTMNonlinearKernel(context,
                          input_row,
                          prev_row,
                          scale_data,
                          params_data,
                          embed_scales,
                          params_stride,
                          cell_dim,
                          output_cell,
                          output_row);
    }

    return MaceStatus::MACE_SUCCESS;
  }

 protected:
  MACE_OP_INPUT_TAGS(INPUT, PARAMS);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterLSTMNonlinear(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "LSTMNonlinear", LSTMNonlinearOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
