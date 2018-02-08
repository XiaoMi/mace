//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_BUFFER_TO_IMAGE_H_
#define MACE_OPS_BUFFER_TO_IMAGE_H_

#include "mace/core/operator.h"
#include "mace/kernels/buffer_to_image.h"

namespace mace {

template <DeviceType D, typename T>
class BufferToImageOp: public Operator<D, T> {
 public:
  BufferToImageOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws)  {}

  bool Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);

    kernels::BufferType type = static_cast<kernels::BufferType>(OperatorBase::GetSingleArgument<int>(
        "buffer_type", static_cast<int>(kernels::CONV2D_FILTER)));
    Tensor *output = this->Output(OUTPUT);

    functor_(const_cast<Tensor *>(input_tensor), type, output, future);
    return true;
  }

 private:
  kernels::BufferToImageFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace mace
#endif  // MACE_OPS_BUFFER_TO_IMAGE_H_
