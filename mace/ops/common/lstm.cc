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

// Details are in
// http://kaldi-asr.org/doc/nnet-simple-component_8h_source.html#l02164

#include "mace/ops/common/lstm.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

void LSTMNonlinearKernel(const OpContext *context,
                         const float *input_data,
                         const float *prev_data,
                         const float *scale_data,
                         const float *params_data,
                         bool embed_scales,
                         index_t params_stride,
                         index_t cell_dim,
                         float *output_cell,
                         float *output_data) {
  float i_scale = (embed_scales && scale_data) ? scale_data[0] : 1.0f;
  float f_scale = (embed_scales && scale_data) ? scale_data[1] : 1.0f;
  float o_scale = (embed_scales && scale_data) ? scale_data[2] : 1.0f;

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
    if (prev_data == nullptr) {
      for (index_t c = start; c < end; c += step) {
        float i_part = input_data[c];
        float c_part = input_data[c + 2 * cell_dim];
        float o_part = input_data[c + 3 * cell_dim];
        float w_oc = params_data[c + params_stride * 2];
        float i_t = ScalarSigmoid(i_part);
        float c_t = i_t * i_scale * std::tanh(c_part);
        float o_t = ScalarSigmoid(o_part + w_oc * c_t);
        float m_t = o_t * o_scale * std::tanh(c_t);
        output_cell[c] = c_t;
        output_data[c] = m_t;
      }
    } else {
      for (index_t c = start; c < end; c += step) {
        float i_part = input_data[c];
        float f_part = input_data[c + cell_dim];
        float c_part = input_data[c + 2 * cell_dim];
        float o_part = input_data[c + 3 * cell_dim];
        float c_prev = prev_data[c];
        float w_ic = params_data[c];
        float w_fc = params_data[c + params_stride];
        float w_oc = params_data[c + params_stride * 2];
        float i_t = ScalarSigmoid(i_part + w_ic * c_prev);
        float f_t = ScalarSigmoid(f_part + w_fc * c_prev);
        float c_t =
            f_t * f_scale * c_prev + i_t * i_scale * std::tanh(c_part);
        float o_t = ScalarSigmoid(o_part + w_oc * c_t);
        float m_t = o_t * o_scale * std::tanh(c_t);
        output_cell[c] = c_t;
        output_data[c] = m_t;
      }
    }
  }, 0, cell_dim, 1);
}

}  // namespace ops
}  // namespace mace
