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

#ifndef MACE_OPS_GLOBAL_AVG_POOLING_H_
#define MACE_OPS_GLOBAL_AVG_POOLING_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/global_avg_pooling.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class GlobalAvgPoolingOp : public Operator<D, T> {
 public:
  GlobalAvgPoolingOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> output_shape(4);
    output_shape[0] = input->shape()[0];
    output_shape[1] = input->shape()[1];
    output_shape[2] = output_shape[3] = 1;

    output->Resize(output_shape);

    auto pooling_func = kernels::GlobalAvgPoolingFunctor<D, T>();
    pooling_func(input->data<float>(), input->shape().data(),
                 output->mutable_data<float>(), future);
    return true;
  }

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_GLOBAL_AVG_POOLING_H_
