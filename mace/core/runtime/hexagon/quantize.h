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

#ifndef MACE_CORE_RUNTIME_HEXAGON_QUANTIZE_H_
#define MACE_CORE_RUNTIME_HEXAGON_QUANTIZE_H_

#include "mace/core/tensor.h"

namespace mace {

class Quantizer {
 public:
  Quantizer() {}
  ~Quantizer() {}

  void Quantize(const Tensor &in_tensor,
                Tensor *out_tensor,
                float *min_out,
                float *max_out);
  void Quantize(const Tensor &in_tensor,
                const float min_in,
                const float max_in,
                Tensor *out_tensor,
                float *min_out,
                float *max_out);
  void DeQuantize(const Tensor &in_tensor,
                  const float min_in,
                  const float max_in,
                  Tensor *out_tensor);

 private:
  void QuantizeAdjustRange(float min_in,
                           float max_in,
                           float *min_out,
                           float *max_out,
                           float *stepsize,
                           float *recip_stepsize);

  MACE_DISABLE_COPY_AND_ASSIGN(Quantizer);
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_HEXAGON_QUANTIZE_H_
