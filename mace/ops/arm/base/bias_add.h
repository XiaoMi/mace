// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_OPS_ARM_BASE_BIAS_ADD_H_
#define MACE_OPS_ARM_BASE_BIAS_ADD_H_

#include "mace/ops/delegator/bias_add.h"

namespace mace {
namespace ops {
namespace arm {

template<typename T>
class BiasAdd : public delegator::BiasAdd {
 public:
  explicit BiasAdd(const DelegatorParam &param) : delegator::BiasAdd(param) {}
  ~BiasAdd() = default;

  MaceStatus Compute(const OpContext *context,
                     const Tensor *input,
                     const Tensor *bias,
                     Tensor *output,
                     const bool isNCHW = true) override;

 private:
  void AddBias(const OpContext *context,
               const Tensor *input,
               const Tensor *bias,
               Tensor *output,
               const bool isNCHW = true);

  template <int Dim>
  void AddBiasNCHW(utils::ThreadPool *thread_pool,
                   const Tensor *input,
                   const Tensor *bias,
                   Tensor *output);
  template <int Dim>
  void AddBiasNHWC(utils::ThreadPool *thread_pool,
                   const Tensor *input,
                   const Tensor *bias,
                   Tensor *output);
};

template <int Dim>
inline index_t bias_index(index_t offset, index_t channel) {
    return offset + channel;
}

template <>
inline index_t bias_index<1>(index_t offset, index_t channel) {
    MACE_UNUSED(offset);
    return channel;
}
}  // namespace arm
}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_ARM_BASE_BIAS_ADD_H_
