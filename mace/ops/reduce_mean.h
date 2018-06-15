//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_REDUCE_MEAN_H_
#define MACE_OPS_REDUCE_MEAN_H_

#include <string>
#include <vector>

#include "mace/core/operator.h"
#include "mace/kernels/reduce_mean.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class ReduceMeanOp : public Operator<D, T> {
 public:
  ReduceMeanOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws),
        functor_(OperatorBase::GetRepeatedArgs<int>("axis"),
                 OperatorBase::GetOptionalArg<bool>("keepdims", false)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    const std::vector<int> axis =
        OperatorBase::GetRepeatedArgs<int>("axis");
    const int left = static_cast<int>(input->dim_size() * -1);
    const int right = static_cast<int>(input->dim_size());
    if (axis.size()) {
      for (unsigned int i = 0; i < axis.size(); ++i) {
        MACE_CHECK(axis[i] > left && axis[i] < right, "Axis is over range.");
      }
    }
    Tensor *output = this->Output(OUTPUT);

    return functor_(input, output, future);
  }

 private:
  kernels::ReduceMeanFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_REDUCE_MEAN_H_
