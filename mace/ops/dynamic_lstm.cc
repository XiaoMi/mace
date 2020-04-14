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

// This Op is for Fused-LstmNonlinearityComponent
// with prev cell states as inputs in Kaldi.
// prev_out_delay: is the IfDefined componnet's delay value.
//                 means which previous frame's output will
//                 be used here as an input.
// prev_cell_delay: similar as prev_out_delay.
// prev_out_offset: output offset.
// prev_out_dim: prev output's dim.
// prev_cell_dim: prev cell's dim.
// bias_a: the first affine's bias' flag, 1:has bias; 0:no bias.
// bias_b: similar to bias_a.
// scale: scale value of previous output and cell.
// forward_indexes: contains the index of frames will be used for computaion.
//                  This is pre-computed in kaldi-onnx converter
// cell_cache_indexes: indicates which frame's cell will be cached for next
//                     computation.
// out_cache_indexes: similar to cell_cache_indexes.
// http://kaldi-asr.org/doc/nnet-combined-component_8h_source.html#l00255
// More details are in docs/development/dynamic_lstm.md

#include <functional>
#include <memory>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/common/lstm.h"
#include "mace/ops/delegator/gemv.h"

#ifdef MACE_ENABLE_NEON
#include <arm_neon.h>
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
        scale_(Operation::GetOptionalArg<float>("scale", 1.0f)),
        subsample_factor_(
            Operation::GetOptionalArg<int>("subsample_factor", 1)),
        forward_indexes_(
            Operation::GetRepeatedArgs<index_t>("forward_indexes")),
        cell_cache_indexes_(
            Operation::GetRepeatedArgs<index_t>("cell_cache_indexes")),
        out_cache_indexes_(
            Operation::GetRepeatedArgs<index_t>("out_cache_indexes")),
        gemv_(delegator::Gemv::Create(
            context->workspace(),
            MACE_DELEGATOR_KEY(Gemv, DeviceType::CPU, T, kCpuImplType),
            DelegatorParam())) {}

  inline void Validate() {
    const Tensor *input = this->Input(0);
    const unsigned int rank = static_cast<unsigned int>(input->dim_size());
    MACE_CHECK(rank >= 2, "DynamicLSTM's input should have at least 2 dims.");
    const index_t input_chunk = input->dim(rank - 2);
    for (size_t i = 0; i < forward_indexes_.size(); ++i) {
      MACE_CHECK(forward_indexes_[i] < input_chunk && forward_indexes_[i] >= 0,
                 "index is over range.");
    }

    MACE_CHECK(this->InputSize() >= 6,
               "DynamicLSTM should have at least six inputs.",
               "But has only ", this->InputSize(), " inputs.");
    MACE_CHECK(prev_cell_delay_ < 0 && prev_out_delay_ < 0,
               "prev_cell_delay(", prev_cell_delay_,
               ") and prev_out_delay(", prev_out_delay_,
               ") should be less than zero.");
    MACE_CHECK(prev_cell_delay_ % subsample_factor_ == 0 &&
        prev_out_delay_ % subsample_factor_ == 0,
               "prev_cell_delay(", prev_cell_delay_,
               ") and prev_out_delay(", prev_out_delay_,
               ") should be multiples of subsample_factor(",
               subsample_factor_, ").");
    MACE_CHECK(prev_out_dim_ > 0 && prev_cell_dim_ > 0,
               "prev_out_dim(", prev_out_dim_,
               ") and prev_cell_dim(", prev_cell_dim_,
               ") should be greater than zero.");
  }

  void UpdateCell(T *cell_data,
                  const index_t cell_dim,
                  const float scale) {
    if (std::abs(scale - 1.f) < 1e-6)
      return;
    const index_t rounds = cell_dim / 4;
    for (index_t i = 0; i < rounds * 4; i += 4) {
#if defined(MACE_ENABLE_NEON) and not defined(MACE_ENABLE_BFLOAT16)
      float32x4_t in_vec = vld1q_f32(cell_data + i);
      float32x4_t scale_vec = vdupq_n_f32(scale);
      in_vec = vmulq_f32(in_vec, scale_vec);
      vst1q_f32(cell_data + i, in_vec);
#else
      for (index_t j = 0; j < 4; ++j) {
        cell_data[i + j] *= scale;
      }
#endif
    }
    for (index_t i = rounds * 4; i < cell_dim; ++i) {
      cell_data[i] *= scale;
    }
  }

  void CopyAndUpdateCell(T *src_data,
                         const index_t cell_dim,
                         const float scale,
                         T *cell_data) {
    if (std::abs(scale - 1.f) < 1e-6) {
      memcpy(cell_data, src_data, cell_dim * sizeof(T));
      return;
    }

    const index_t rounds = cell_dim / 4;
    for (index_t i = 0; i < rounds * 4; i += 4) {
#if defined(MACE_ENABLE_NEON) and not defined(MACE_ENABLE_BFLOAT16)
      float32x4_t in_vec = vld1q_f32(src_data + i);
      float32x4_t scale_vec = vdupq_n_f32(scale);
      in_vec = vmulq_f32(in_vec, scale_vec);
      vst1q_f32(cell_data + i, in_vec);
#else
      for (index_t j = 0; j < 4; ++j) {
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
    Validate();
    const Tensor *input = this->Input(INPUT);
    const Tensor *prev_out = this->Input(PREV_OUT);
    const Tensor *prev_cell = this->Input(PREV_CELL);
    const Tensor *weights_a = this->Input(WEIGHTS_A);
    const Tensor *lstm_params = this->Input(PARAMS);
    const Tensor *weights_b = this->Input(WEIGHTS_B);
    int max_input_num = 6;
    max_input_num = has_bias_a_ ? max_input_num + 1 : max_input_num;
    MACE_CHECK(this->InputSize() >= max_input_num,
               "The first affine needs a bias input.");
    const Tensor *bias_a = has_bias_a_ ?
                           this->Input(max_input_num - 1) :
                           nullptr;
    max_input_num = has_bias_b_ ? max_input_num + 1 : max_input_num;
    MACE_CHECK(this->InputSize() >= max_input_num,
               "The second affine needs a bias input.");
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
    MACE_CHECK(lstm_input_dim == (lstm_cell_dim * 5),
               "lstm_input_dim(", lstm_input_dim,
               ") should be 5 times of lstm_cell_dim(",
               lstm_cell_dim, ").");
    MACE_CHECK(lstm_params->dim(0) == 3 &&
        params_stride == lstm_cell_dim && lstm_cell_dim == prev_cell_dim_)
        << " lstm params rows: " << lstm_params->dim(0)
        << " params_stride: " << params_stride
        << " != " << " cell_dim: " << lstm_cell_dim << std::endl;
    const index_t affine_b_out_dim = weights_b->dim(0);
    const index_t affine_b_depth = weights_b->dim(1);
    const index_t affine_b_in_dim = lstm_cell_dim;
    MACE_CHECK(affine_b_in_dim == affine_b_depth)
        << "affine_b's input_dim:" << affine_b_in_dim
        << "!=" << "affine_b's weights' depth:" << affine_b_depth << std::endl;

    const index_t output_dim = affine_b_out_dim;
    MACE_CHECK(prev_out_offset_ + prev_out_dim_ <= output_dim)
        << " prev_out_offset: " << prev_out_offset_
        << " prev_out_dim: " << prev_out_dim_
        << " output_dim: " << output_dim;

    const index_t affine_a_in_size =
        PadAlignSize(affine_a_in_dim * sizeof(T));
    const index_t affine_a_out_size =
        PadAlignSize(affine_a_out_dim * sizeof(T));
    const index_t affine_b_in_size =
        PadAlignSize(affine_b_in_dim * sizeof(T));
    const index_t affine_b_out_size =
        PadAlignSize(affine_b_out_dim * sizeof(T));

    const int out_buf_chunk = abs(prev_out_delay_ / subsample_factor_);
    const int cell_buf_chunk = abs(prev_cell_delay_ / subsample_factor_);
    const index_t out_buf_size =
        PadAlignSize(out_buf_chunk * prev_out_dim_ * sizeof(T));
    const index_t cell_buf_size =
        PadAlignSize(cell_buf_chunk * prev_cell_dim_ * sizeof(T));
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    scratch->GrowSize(affine_a_in_size + affine_a_out_size
                          + affine_b_in_size + affine_b_out_size
                          + out_buf_size + cell_buf_size);

    Tensor prev_out_buf(scratch->Scratch(out_buf_size), DataTypeToEnum<T>::v());
    prev_out_buf.Reshape({out_buf_chunk, prev_out_dim_});
    T *prev_out_buf_data = prev_out_buf.mutable_data<T>();

    Tensor prev_cell_buf(
        scratch->Scratch(cell_buf_size), DataTypeToEnum<T>::v());
    prev_cell_buf.Reshape({cell_buf_chunk, prev_cell_dim_});
    T *prev_cell_buf_data = prev_cell_buf.mutable_data<T>();

    Tensor affine_a_in(
        scratch->Scratch(affine_a_in_size), DataTypeToEnum<T>::v());
    affine_a_in.Reshape({1, affine_a_in_dim});
    T *affine_a_in_data = affine_a_in.mutable_data<T>();

    Tensor affine_a_out(
        scratch->Scratch(affine_a_out_size), DataTypeToEnum<T>::v());
    affine_a_out.Reshape({1, affine_a_out_dim});
    T *affine_a_out_data = affine_a_out.mutable_data<T>();

    Tensor affine_b_in(
        scratch->Scratch(affine_b_in_size), DataTypeToEnum<T>::v());
    affine_b_in.Reshape({1, affine_b_in_dim});
    T *affine_b_in_data = affine_b_in.mutable_data<T>();

    Tensor affine_b_out(
        scratch->Scratch(affine_b_out_size), DataTypeToEnum<T>::v());
    affine_b_out.Reshape({1, affine_b_out_dim});
    T *affine_b_out_data = affine_b_out.mutable_data<T>();

    Tensor *output = this->Output(OUTPUT);
    Tensor *out_cache = this->Output(OUT_CACHE);
    Tensor *cell_cache = this->Output(CELL_CACHE);

    std::vector<index_t> output_shape = input->shape();
    const index_t out_chunk = forward_indexes_.size();
    output_shape[input_rank - 1] = output_dim;
    std::vector<index_t> prev_out_shape = input->shape();
    prev_out_shape[input_rank - 1] = prev_out_dim_;
    prev_out_shape[input_rank - 2] = out_buf_chunk;
    std::vector<index_t> prev_cell_shape = input->shape();
    prev_cell_shape[input_rank - 1] = prev_cell_dim_;
    prev_cell_shape[input_rank - 2] = cell_buf_chunk;

    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    MACE_RETURN_IF_ERROR(out_cache->Resize(prev_out_shape));
    MACE_RETURN_IF_ERROR(cell_cache->Resize(prev_cell_shape));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard prev_out_guard(prev_out);
    Tensor::MappingGuard prev_cell_guard(prev_cell);
    Tensor::MappingGuard lstm_params_guard(lstm_params);

    Tensor::MappingGuard output_guard(output);
    Tensor::MappingGuard out_cache_guard(out_cache);
    Tensor::MappingGuard cell_cache_guard(cell_cache);

    const T *input_data = input->data<T>();
    const T *prev_out_data = prev_out->data<T>();
    const T *prev_cell_data = prev_cell->data<T>();
    const T *lstm_params_data = lstm_params->data<T>();
    T *output_data = output->mutable_data<T>();
    T *out_cache_data = out_cache->mutable_data<T>();
    T *cell_cache_data = cell_cache->mutable_data<T>();

    for (int b = 0; b < batch; ++b) {
      memcpy(prev_out_buf_data,
             prev_out_data + b * out_buf_chunk * prev_out_dim_,
             sizeof(T) * out_buf_chunk * prev_out_dim_);
      memcpy(prev_cell_buf_data,
             prev_cell_data + b * cell_buf_chunk * prev_cell_dim_,
             sizeof(T) * cell_buf_chunk * prev_cell_dim_);

      for (index_t i = 0; i < out_chunk; ++i) {
        const T *input_ptr =
            input_data + (b * chunk + forward_indexes_[i]) * input_dim;
        T *output_ptr = output_data + (b * out_chunk + i) * output_dim;
        // Append
        memcpy(affine_a_in_data, input_ptr, input_dim * sizeof(T));
        memcpy(affine_a_in_data + input_dim,
               prev_out_buf_data + i % out_buf_chunk * prev_out_dim_,
               prev_out_dim_ * sizeof(T));
        // Affine
        gemv_->Compute(context,
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
        T *lstm_cell_ptr =
            prev_cell_buf_data + i % cell_buf_chunk * prev_cell_dim_;
        T *curr_cell_ptr = lstm_cell_ptr;
        // LSTMNonlinear
        LSTMNonlinearKernel<T>(context,
                            affine_a_out_data,
                            lstm_cell_ptr,
                            nullptr,
                            lstm_params_data,
                            false,
                            params_stride,
                            lstm_cell_dim,
                            curr_cell_ptr,
                            affine_b_in_data);
        UpdateCell(curr_cell_ptr, prev_cell_dim_, scale_);
        // Affine
        gemv_->Compute(context,
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
               output_dim * sizeof(T));
        // Update
        T *curr_out_ptr =
            prev_out_buf_data + i % out_buf_chunk * prev_out_dim_;
        CopyAndUpdateCell(affine_b_out_data + prev_out_offset_,
                          prev_out_dim_,
                          scale_,
                          curr_out_ptr);

        for (size_t k = 0; k < out_cache_indexes_.size(); ++k) {
          if (i == out_cache_indexes_[k]) {
            const index_t idx = b * out_buf_chunk + k;
            T *out_cache_ptr =
                out_cache_data + idx * prev_out_dim_;
            memcpy(out_cache_ptr,
                   curr_out_ptr,
                   sizeof(T) * prev_out_dim_);
          }
        }

        for (size_t k = 0; k < cell_cache_indexes_.size(); ++k) {
          if (i == cell_cache_indexes_[k]) {
            const index_t idx = b * cell_buf_chunk + k;
            T *cell_cache_ptr =
                cell_cache_data + idx * prev_cell_dim_;
            memcpy(cell_cache_ptr,
                   curr_cell_ptr,
                   sizeof(T) * prev_cell_dim_);
          }
        }
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
  int subsample_factor_;
  std::vector<index_t> forward_indexes_;
  std::vector<index_t> cell_cache_indexes_;
  std::vector<index_t> out_cache_indexes_;
  std::unique_ptr<delegator::Gemv> gemv_;

  MACE_OP_INPUT_TAGS(INPUT, PREV_OUT, PREV_CELL, WEIGHTS_A, PARAMS, WEIGHTS_B);
  MACE_OP_OUTPUT_TAGS(OUTPUT, OUT_CACHE, CELL_CACHE);
};

void RegisterDynamicLSTM(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "DynamicLSTM", DynamicLSTMOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "DynamicLSTM", DynamicLSTMOp,
                        DeviceType::CPU);
}

}  // namespace ops
}  // namespace mace
