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

#ifndef MACE_OPS_TRANSPOSE_H_
#define MACE_OPS_TRANSPOSE_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/softmax.h"
#include "mace/kernels/transpose.h"

namespace mace {

template <DeviceType D, class T>
class TransposeOp : public Operator<D, T> {
 public:
  TransposeOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        dims_(OperatorBase::GetRepeatedArgs<int>("dims")),
        functor_(dims_) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    const std::vector<index_t> &input_shape = input->shape();
    MACE_CHECK((input_shape.size() == 4 && dims_.size() == 4) ||
                   (input_shape.size() == 2 && dims_.size() == 2),
               "rank should be 2 or 4");
    std::vector<index_t> output_shape;
    for (size_t i = 0; i < dims_.size(); ++i) {
      output_shape.push_back(input_shape[dims_[i]]);
    }
    MACE_RETURN_IF_ERROR(output->Resize(output_shape));
    return functor_(input, output, future);
  }

 protected:
  std::vector<int> dims_;
  kernels::TransposeFunctor<D, T> functor_;

  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_TRANSPOSE_H_
