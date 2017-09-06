//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/ops/conv_2d.h"
#include "mace/proto/mace.pb.h"

namespace mace {

template <>
bool Conv2dOp<DeviceType::CPU, float>::Run() {
  const Tensor* input = Input(INPUT);
  const Tensor* filter = Input(FILTER);
  const Tensor* bias = Input(BIAS);
  Tensor* output = Output(OUTPUT);


  // Test
  VLOG(0) << "conv_2d([" << kernels_[0] << ", " << kernels_[1]  << "], )";
  const float* input_data = input->data<float>();
  for (int i = 0; i < 6; ++i) {
    VLOG(0) << input_data[i];
  }

  return true;
}


REGISTER_CPU_OPERATOR(Conv2d, Conv2dOp<DeviceType::CPU, float>);

}
