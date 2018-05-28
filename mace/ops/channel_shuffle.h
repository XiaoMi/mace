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

#ifndef MACE_OPS_CHANNEL_SHUFFLE_H_
#define MACE_OPS_CHANNEL_SHUFFLE_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/channel_shuffle.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class ChannelShuffleOp : public Operator<D, T> {
 public:
  ChannelShuffleOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        group_(OperatorBase::GetOptionalArg<int>("group", 1)),
        functor_(this->group_) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    int channels;
    if (D == GPU) {
      channels = input->dim(3);
    } else if (D == CPU) {
      channels = input->dim(1);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
    MACE_CHECK(channels % group_ == 0,
               "input channels must be an integral multiple of group. ",
               input->dim(3));
    return functor_(input, output, future);
  }

 protected:
  const int group_;
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::ChannelShuffleFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CHANNEL_SHUFFLE_H_
