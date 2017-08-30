//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/relu.h"
#include "mace/proto/mace.pb.h"

namespace mace {

template <>
bool ReluOp<DeviceType::CPU, float>::Run() {
  const Tensor* X = Input(0);
  Tensor* Y = Output(0);
  Y->ResizeLike(X);

  const float* Xdata = X-> data<float>();
  float* Ydata = Y->mutable_data<float>();
  for (int i = 0; i < X->size(); ++i) {
    Ydata[i] = std::max(Xdata[i], 0.f);
    VLOG(0) << i << ": " << Xdata[i] << " " << Ydata[i];
  }

  return true;
}

REGISTER_CPU_OPERATOR(Relu, ReluOp<DeviceType::CPU, float>);

} //  namespace mace
