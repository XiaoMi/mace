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

#ifndef MACE_OPS_CAST_H_
#define MACE_OPS_CAST_H_

#include <vector>

#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, typename SrcType>
class CastOp : public Operator<D, SrcType> {
 public:
  CastOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, SrcType>(op_def, ws) {}

  MaceStatus Run(StatsFuture *future) override {
    MACE_UNUSED(future);
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
    MACE_RETURN_IF_ERROR(output->ResizeLike(input))

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto dst_dtype = output->dtype();

#define MACE_CAST_COPY \
    auto output_data = output->mutable_data<T>();                       \
    auto input_data = input->data<SrcType>();                           \
    for (index_t i = 0; i < output->size(); ++i) {                      \
      output_data[i] = static_cast<T>(input_data[i]);                   \
    }

    MACE_RUN_WITH_TYPE_ENUM(dst_dtype, MACE_CAST_COPY);

    return MACE_SUCCESS;
  }

 private:
  MACE_OP_INPUT_TAGS(INPUT);
  MACE_OP_OUTPUT_TAGS(OUTPUT);
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CAST_H_
