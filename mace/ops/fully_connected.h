//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_FULLY_CONNECTED_H_
#define MACE_OPS_FULLY_CONNECTED_H_

#include <string>

#include "mace/core/operator.h"
#include "mace/kernels/fully_connected.h"

namespace mace {

template <DeviceType D, class T>
class FullyConnectedOp : public Operator<D, T> {
 public:
  FullyConnectedOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(static_cast<kernels::BufferType>(
                     OperatorBase::GetSingleArgument<int>(
                         "weight_type", static_cast<int>(
                             kernels::WEIGHT_WIDTH))),
                 kernels::StringToActivationType(
                     OperatorBase::GetSingleArgument<std::string>("activation",
                                                                  "NOOP")),
                 OperatorBase::GetSingleArgument<float>("max_limit", 0.0f)) {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const Tensor *weight = this->Input(WEIGHT);
    const Tensor *bias = this->InputSize() >= 3 ? this->Input(BIAS) : nullptr;
    Tensor *output = this->Output(OUTPUT);

    const index_t input_size = input->dim(1) * input->dim(2) * input->dim(3);
    MACE_CHECK(input_size == weight->dim(1) && weight->dim(0) == bias->dim(0))
        << "The size of Input, Weight and Bias don't match.";

    functor_(input, weight, bias, output, future);
    return true;
  }

 private:
  kernels::FullyConnectedFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_FULLY_CONNECTED_H_
