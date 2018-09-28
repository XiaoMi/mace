// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_OPS_DECONV_2D_H_
#define MACE_OPS_DECONV_2D_H_

#include "mace/core/types.h"

namespace mace {
namespace ops {

enum FrameworkType {
  TENSORFLOW = 0,
  CAFFE = 1,
};

template <typename T>
void CropPadOut(const T *input,
                const index_t *in_shape,
                const index_t *out_shape,
                const index_t pad_h,
                const index_t pad_w,
                T *output) {
  const index_t batch = in_shape[0];
  const index_t channel = in_shape[1];
  const index_t in_height = in_shape[2];
  const index_t in_width = in_shape[3];

  const index_t out_height = out_shape[2];
  const index_t out_width = out_shape[3];
#pragma omp parallel for collapse(3)
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channel; ++j) {
      for (int k = 0; k < out_height; ++k) {
        const T *input_base =
            input + ((i * channel + j) * in_height + (k + pad_h)) * in_width;
        T *output_base =
            output + ((i * channel + j) * out_height + k)* out_width;
        memcpy(output_base, input_base + pad_w, out_width * sizeof(T));
      }
    }
  }
}

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_DECONV_2D_H_
