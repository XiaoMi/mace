//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_IMAGE_TO_BUFFER_H_
#define MACE_OPS_IMAGE_TO_BUFFER_H_

#include "mace/core/operator.h"
#include "mace/kernels/buffer_to_image.h"

namespace mace {

template <DeviceType D, typename T>
class ImageToBufferOp: public Operator<D, T> {
 public:
  ImageToBufferOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws), functor_(true)  {}

  bool Run() override {
    const Tensor *input_tensor = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);

    kernels::BufferType type = static_cast<kernels::BufferType>(OperatorBase::GetSingleArgument<int>(
        "buffer_type", static_cast<int>(kernels::FILTER)));
    functor_(output, type, const_cast<Tensor *>(input_tensor));
    return true;
  }

 private:
  kernels::BufferToImageFunctor<D, T> functor_;

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);
};

} //  namespace mace
#endif //  MACE_OPS_IMAGE_TO_BUFFER_H_
