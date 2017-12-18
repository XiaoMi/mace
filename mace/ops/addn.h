//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_ADDN_H_
#define MACE_OPS_ADDN_H_

#include "mace/core/operator.h"
#include "mace/kernels/addn.h"

namespace mace {

template <DeviceType D, class T>
class AddNOp : public Operator<D, T> {
 public:
  AddNOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}

  bool Run(StatsFuture *future) override {
    Tensor *output_tensor = this->outputs_[0];
    int n = this->inputs_.size();
    vector<const Tensor *> inputs(n, nullptr);
    for (int i = 0; i < n; ++i) {
      inputs[i] = this->inputs_[i];
    }

    functor_(inputs, output_tensor, future);
    return true;
  }

 private:
  kernels::AddNFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_ADDN_H_
