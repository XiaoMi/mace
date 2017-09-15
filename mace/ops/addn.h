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
  AddNOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<D, T>(operator_def, ws) {}

  bool Run() override {
    Tensor* output_tensor = this->outputs_[0];
    output_tensor->ResizeLike(this->inputs_[0]);
    T* output = output_tensor->mutable_data<T>();
    index_t size = this->inputs_[0]->size();
    int n = this->inputs_.size();
    vector<const T*> inputs(n);
    for (int i = 0; i < n; ++i) {
      const Tensor* input_tensor = this->inputs_[i];
      inputs[i] = input_tensor->data<T>();
    }

    functor_(inputs, output, size);
    return true;
  }

 private:
  kernels::AddNFunctor<D, T> functor_;
};

}  // namespace mace

#endif  // MACE_OPS_ADDN_H_
