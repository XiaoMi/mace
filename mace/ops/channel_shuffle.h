//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CHANNEL_SHUFFLE_H_
#define MACE_OPS_CHANNEL_SHUFFLE_H_

#include <memory>

#include "mace/core/operator.h"
#include "mace/kernels/channel_shuffle.h"

namespace mace {

template <DeviceType D, typename T>
class ChannelShuffleOp : public Operator<D, T> {
 public:
  ChannelShuffleOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        group_(OperatorBase::GetSingleArgument<int>("group", 1)),
        functor_(this->group_) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_CHECK(input->shape()[1] % group_ == 0,
               "input channels must be an integral multiple of group. ",
               input->shape()[1]);

    output->ResizeLike(input);
    functor_(input->data<T>(), input->shape().data(),
             output->mutable_data<T>(), future);

    return true;
  }

 protected:
  const int group_;
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::ChannelShuffleFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_CHANNEL_SHUFFLE_H_
