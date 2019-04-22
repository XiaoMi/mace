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

// This Op is for Fused-LstmNonlinearComponent
// with prev cell states as inputs in Kaldi.
// http://kaldi-asr.org/doc/nnet-simple-component_8h_source.html#l02164
// More details are in docs/development/dynamic_lstm.md

#include <functional>
#include <memory>

#include "mace/core/operator.h"
#include "mace/ops/common/lstm.h"

#ifdef MACE_ENABLE_NEON
#include <arm_neon.h>
#include "mace/ops/arm/fp32/gemv.h"
#else
#include "mace/ops/ref/gemv.h"
#endif  // MACE_ENABLE_NEON

namespace mace {
namespace ops {

template<DeviceType D, typename T>
class DynamicLSTMOp;

template<typename T>
class DynamicLSTMOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit DynamicLSTMOp(OpConstructContext *context)
      : Operation(context),
        prev_out_delay_(
            Operation::GetOptionalArg<int>("prev_out_delay", 0)),
        prev_cell_delay_(
            Operation::GetOptionalArg<int>("prev_cell_delay", 0)),
        prev_out_offset_(Operation::GetOptionalArg<int>("prev_out_offset", 0)),
        prev_out_dim_(Operation::GetOptionalArg<int>("prev_out_dim", 0)),
        prev_cell_dim_(Operation::GetOptionalArg<int>("prev_cell_dim", 0)),
        has_bias_a_(Operation::GetOptionalArg<int>("bias_a", 1)),
        has_bias_b_(Operation::GetOptionalArg<int>("bias_b", 1)),
        scale_(Operation::GetOptionalArg<float>("scale", 1.0f)) {}

  void UpdateCell(float *cell_data,
                  const index_t cell_dim,
                  const float scale) {
    if (std::abs(scale - 1.f) < 1e-6)
      return;
    const index_t rounds = cell_dim / 4;
    for (index_t i = 0; i < rounds * 4; i += 4) {
#ifdef MACE_ENABLE_NEON
      float32x4_t in_vec = vld1q_f32(cell_data + i);
      float32x4_t scale_vec = vdupq_n_f32(scale);
      in_vec = vmulq_f32(in_vec, scale_vec);
      vst1q_f32(cell_data + i, in_vec);
#else
      for (int j = 0; j < 4; ++j) {
        cell_data[i + j] *= scale;
      }
#endif
    }
    for (index_t i = rounds * 4; i < cell_dim; ++i) {
      cell_data[i] *= scale;
    }
  }

  void CopyAndUpdateCell(float *src_data,
                         const index_t cell_dim,
                         const float scale,
                         float *cell_data) {
    if (std::abs(scale - 1.f) < 1e-6) {
      memcpy(cell_data, src_data, cell_dim * sizeof(float));
      return;
    }

    const index_t rounds = cell_dim / 4;
    for (index_t i = 0; i < rounds * 4; i += 4) {
#ifdef MACE_ENABLE_NEON
      float32x4_t in_vec = vld1q_f32(src_data + i);
      float32x4_t scale_vec = vdupq_n_f32(scale);
      in_vec = vmulq_f32(in_vec, scale_vec);
      vst1q_f32(cell_data + i, in_vec);
#else
      for (int j = 0; j < 4; ++j) {
        cell_data[i + j] = src_data[i + j] * scale;
      }
#endif
    }
    for (index_t i = rounds * 4; i < cell_dim; ++i) {
      cell_data[i] = src_data[i] * scale;
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    int max_input_num = 4;
    MACE_CHECK(this->InputSize() >= max_input_num,
               "DynamicLSTM has at least four inputs.");
    MACE_CHECK(prev_cell_delay_ < 0 && prev_out_delay_ < 0);
    MACE_CHECK(prev_out_dim_ > 0 && prev_cell_dim_ > 0);
    const Tensor *input = this->Input(INPUT);
    const Tensor *weights_a = this->Input(WEIGHTS_A);
    const Tensor *lstm_params = this->Input(PARAMS);
    const Tensor *weights_b = this->Input(WEIGHTS_B);
    if (has_bias_a_) {
      max_input_num++;
      MACE_CHECK(this->InputSize() >= max_input_num,
                 "The first affine needs a bias input.");
    }
    const Tensor *bias_a = has_bias_a_ ?
                           this->Input(max_input_num - 1) :
                           nullptr;
    if (has_bias_b_) {
      max_input_num++;
      MACE_CHECK(this->InputSize() >= max_input_num,
                 "The second affine needs a bias input.");
    }
    const Tensor *bias_b = has_bias_b_ ?
                           this->Input(max_input_num - 1) :
                           nullptr;

    const index_t input_rank = input->dim_size();
    MACE_CHECK(input_rank >= 2,
               "Dynamic LSTM Cell's input dim size should be >= 2.");
    const std::vector<index_t> &input_shape = input->shape();
    const index_t batch =
        std::accumulate(input_shape.begin(), input_shape.end() - 2, 1,
                        std::multiplies<index_t>());
    const index_t chunk = input_shape[input_rank - 2];
    const index_t input_dim = input_shape[input_rank - 1];

    const index_t affine_a_in_dim = input_dim + prev_out_dim_;
    const index_t affine_a_out_dim = weights_a->dim(0);
    const index_t affine_a_depth = weights_a->dim(1);
    MACE_CHECK(affine_a_in_dim == affine_a_depth)
      << "affine_a's input_dim:" << affine_a_in_dim
      << "!=" << "affine_a's weights' depth:" << affine_a_depth << std::endl;

    const index_t lstm_input_dim = affine_a_out_dim + prev_cell_dim_;
    const index_t lstm_cell_dim = lstm_input_dim / 5;
    const index_t params_stride = lstm_params->dim(1);
    MACE_CHECK(lstm_input_dim == (lstm_cell_dim * 5));
    MACE_CHECK(lstm_params->dim(0) == 3 &&
        params_stride == lstm_cell_dim && lstm_cell_dim == prev_cell_dim_)
      << "lstm params rows:" << lstm_params->dim(0)
      << "params_stride:" << params_stride
      << "!=" << "cell_dim:" << lstm_cell_dim << std::endl;
    const index_t affine_b_out_dim = weights_b->dim(0);
    const index_t affine_b_depth = weights_b->dim(1);
    const index_t affine_b_in_dim = lstm_cell_dim;
    MACE_CHECK(affine_b_in_dim == affine_b_depth)
      << "affine_b's input_dim:" << affine_b_in_dim
      << "!=" << "affine_b's weights' depth:" << affine_b_depth << std::endl;

    const index_t output_dim = affine_b_out_dim;
    MACE_CHECK(prev_out_offset_ + prev_out_dim_ <= output_dim);

    const index_t affine_a_in_size =
        PadAlignSize(affine_a_in_dim * sizeof(float));
    const index_t affine_a_out_size =
        PadAlignSize(affine_a_out_dim * sizeof(float));
    const index_t affine_b_in_size =
        PadAlignSize(affine_b_in_dim * sizeof(float));
    const index_t affine_b_out_size =
        PadAlignSize(affine_b_out_dim * sizeof(float));

    const int out_buf_chunk = abs(prev_out_delay_);
    const int cell_buf_chunk = abs(prev_cell_delay_);
    const index_t out_buf_size =
        PadAlignSize(out_buf_chunk * prev_out_dim_ * sizeof(float));
    const index_t cell_buf_size =
        PadAlignSize(cell_buf_chunk * prev_cell_dim_ * sizeof(float));
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(affine_a_in_size + affine_a_out_size
                          + affine_b_in_size + affine_b_out_size
                          + out_buf_size + cell_buf_size);

    Tensor prev_out(scratch->Scratch(out_buf_size), DT_FLOAT);
    prev_out.Reshape({out_buf_chunk, prev_out_dim_});
    float *prev_out_data = prev_out.mutable_data<float>();

    Tensor prev_cell(scratch->Scratch(cell_buf_size), DT_FLOAT);
    prev_cell.Reshape({cell_buf_chunk, prev_cell_dim_});
    float *prev_cell_data = prev_cell.mutable_data<float>();

    Tensor affine_a_in(scratch->Scratch(affine_a_in_size), DT_FLOAT);
    affine_a_in.Reshape({1, affine_a_in_dim});
    float *affine_a_in_data = affine_a_in.mutable_data<float>();

    Tensor affine_a_out(scratch->Scratch(affine_a_out_size), DT_FLOAT);
    affine_a_out.Reshape({1, affine_a_out_dim});
    float *affine_a_out_data = affine_a_out.mutable_data<float>();

    Tensor affine_b_in(scratch->Scratch(affine_b_in_size), DT_FLOAT);
    affine_b_in.Reshape({1, affine_b_in_dim});
    float *affine_b_in_data = affine_b_in.mutable_data<float>();

    Tensor affine_b_out(scratch->Scratch(affine_b_out_size), DT_FLOAT);
    affine_b_out.Reshape({1, affine_b_out_dim});
    float *affine_b_out_data = affine_b_out.mutable_data<float>();

    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> output_shape = input->shape();
    output_shape[input_rank - 1] = output_dim;

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard lstm_params_guard(lstm_params);
    Tensor::MappingGuard output_guard(output);
    const float *input_data = input->data<float>();
    const float *lstm_params_data = lstm_params->data<float>();
    float *output_data = output->mutable_data<float>();

    for (int b = 0; b < batch; ++b) {
      int prev_out_idx = prev_out_delay_;
      int prev_cell_idx = prev_cell_delay_;
      prev_cell.Clear();
      prev_out.Clear();
      affine_a_in.Clear();
      affine_a_out.Clear();
      affine_b_in.Clear();
      affine_b_out.Clear();
      for (int i = 0; i < chunk; ++i) {
        const float *input_ptr = input_data + (b * chunk + i) * input_dim;
        float *output_ptr = output_data + (b * chunk + i) * output_dim;
        // Append
        memcpy(affine_a_in_data, input_ptr, input_dim * sizeof(float));
        if (prev_out_idx >= 0) {
          memcpy(affine_a_in_data + input_dim,
                 prev_out_data + prev_out_idx % out_buf_chunk * prev_out_dim_,
                 prev_out_dim_ * sizeof(float));
        }
        // Affine
        gemv_.Compute(context,
                      weights_a,
                      &affine_a_in,
                      bias_a,
                      1,
                      affine_a_out_dim,
                      affine_a_depth,
                      false,
                      false,
                      &affine_a_out);
        // Prepare LSTMNonlinear input and output pointer
        float *prev_cell_ptr =
            prev_cell_idx < 0 ? nullptr :
            prev_cell_data + prev_cell_idx % cell_buf_chunk * prev_cell_dim_;
        float *curr_cell_ptr =
            prev_cell_data + i % cell_buf_chunk * prev_cell_dim_;
        // LSTMNonlinear
        LSTMNonlinearKernel(context,
                            affine_a_out_data,
                            prev_cell_ptr,
                            nullptr,
                            lstm_params_data,
                            false,
                            params_stride,
                            lstm_cell_dim,
                            curr_cell_ptr,
                            affine_b_in_data);
        UpdateCell(curr_cell_ptr, prev_cell_dim_, scale_);
        // Affine
        gemv_.Compute(context,
                      weights_b,
                      &affine_b_in,
                      bias_b,
                      1,
                      affine_b_out_dim,
                      affine_b_depth,
                      false,
                      false,
                      &affine_b_out);
        // Output
        memcpy(output_ptr,
               affine_b_out_data,
               output_dim * sizeof(float));
        // Update
        float *curr_out_ptr = prev_out_data + i % out_buf_chunk * prev_out_dim_;
        CopyAndUpdateCell(affine_b_out_data + prev_out_offset_,
                          prev_out_dim_,
                          scale_,
                          curr_out_ptr);
        prev_out_idx++;
        prev_cell_idx++;
      }
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int prev_out_delay_;
  int prev_cell_delay_;
  int prev_out_offset_;
  int prev_out_dim_;
  int prev_cell_dim_;
  int has_bias_a_;
  int has_bias_b_;
  float scale_;

#ifdef MACE_ENABLE_NEON
  arm::fp32::Gemv gemv_;
#else
  ref::Gemv<float> gemv_;
#endif  // MACE_ENABLE_NEON

  MACE_OP_INPUT_TAGS(INPUT, WEIGHTS_A, PARAMS, WEIGHTS_B);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

void RegisterDynamicLSTM(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "DynamicLSTM", DynamicLSTMOp,
                   DeviceType::CPU, float);
}

}  // namespace ops
}  // namespace mace
