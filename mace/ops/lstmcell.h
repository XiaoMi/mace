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

#ifndef MACE_OPS_LSTMCELL_H_
#define MACE_OPS_LSTMCELL_H_

#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/lstmcell.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class LSTMCellOp : public Operator<D, T> {
 public:
  LSTMCellOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(static_cast<T>(
              OperatorBase::GetOptionalArg<float>("value", 0.0))) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *pre_output = this->Input(PRE_OUTPUT);
    const Tensor *weight = this->Input(WEIGHT);
    const Tensor *bias = this->Input(BIAS);
    const Tensor *pre_cell = this->Input(PRE_CELL);
    Tensor *cell = this->Output(CELL);
    Tensor *output = this->Output(OUTPUT);

    MACE_CHECK(input->dim_size() == 2 && input->dim(1) % 4 == 0,
               "LSTM step should be a multiple of 4");

    return functor_(
        input, pre_output, weight, bias, pre_cell, cell, output, future);
  };

 protected:
  kernels::LSTMCellFunctor<D, T> functor_;

  MACE_OP_INPUT_TAGS(INPUT, PRE_OUTPUT, WEIGHT, BIAS, PRE_CELL);
  MACE_OP_OUTPUT_TAGS(CELL, OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_LSTMCELL_H_
