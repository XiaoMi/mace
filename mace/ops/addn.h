//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_ADDN_H_
#define MACE_OPS_ADDN_H_

#include "mace/core/operator.h"
#include "mace/kernels/addn.h"

namespace mace {

template<DeviceType D, class T>
class AddNOp : public Operator<D, T> {
 public:
  AddNOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}
  bool Run() override {
    Tensor* output_tensor = this->Output(0);
    kernels::AddNFuntion<T>(this->Inputs(), output_tensor);
    return true;
  }
};

} // namespace mace

#endif // MACE_OPS_ADDN_H_
