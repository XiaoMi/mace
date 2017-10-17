//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_GLOBAL_AVG_POOLING_H_
#define MACE_OPS_GLOBAL_AVG_POOLING_H_

#include "mace/core/operator.h"
#include "mace/kernels/global_avg_pooling.h"

namespace mace {

template <DeviceType D, class T>
class GlobalAvgPoolingOp : public Operator<D, T> {
 public:
  GlobalAvgPoolingOp(const OperatorDef &operator_def, Workspace *ws)
      : Operator<D, T>(operator_def, ws) {}

  bool Run() override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    std::vector<index_t> output_shape(4);
    output_shape[0] = input->shape()[0];
    output_shape[1] = input->shape()[1];
    output_shape[2] = output_shape[3] = 1;

    output->Resize(output_shape);

    auto pooling_func = kernels::GlobalAvgPoolingFunctor<D, T>();
    pooling_func(input->data<float>(), input->shape().data(),
                 output->mutable_data<float>());
    return true;
  }

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace

#endif  // MACE_OPS_GLOBAL_AVG_POOLING_H_
