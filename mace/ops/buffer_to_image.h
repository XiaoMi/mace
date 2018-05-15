// Copyright 2018 Xiaomi, Inc.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MACE_OPS_BUFFER_TO_IMAGE_H_
#define MACE_OPS_BUFFER_TO_IMAGE_H_

#include "mace/core/operator.h"
#include "mace/kernels/buffer_to_image.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class BufferToImageOp : public Operator<D, T> {
 public:
  BufferToImageOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetOptionalArg<int>("wino_block_size", 2)) {}

  MaceStatus Run(StatsFuture *future) override {
    const Tensor *input_tensor = this->Input(INPUT);

    kernels::BufferType type =
        static_cast<kernels::BufferType>(OperatorBase::GetOptionalArg<int>(
            "buffer_type", static_cast<int>(kernels::CONV2D_FILTER)));
    Tensor *output = this->Output(OUTPUT);

    return functor_(input_tensor, type, output, future);
  }

 private:
  kernels::BufferToImageFunctor<D, T> functor_;

 protected:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace
#endif  // MACE_OPS_BUFFER_TO_IMAGE_H_
