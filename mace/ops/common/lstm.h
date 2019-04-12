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

#ifndef MACE_OPS_COMMON_LSTM_H_
#define MACE_OPS_COMMON_LSTM_H_

#include "mace/core/types.h"
#include "mace/core/op_context.h"

namespace mace {
namespace ops {

void LSTMNonlinearKernel(const OpContext *opContext,
                         const float *input_data,
                         const float *prev_data,
                         const float *scale_data,
                         const float *params_data,
                         bool embed_scales,
                         index_t params_stride,
                         index_t cell_dim,
                         float *output_cell,
                         float *output_data);

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_COMMON_LSTM_H_

