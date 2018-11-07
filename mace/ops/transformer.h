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

#ifndef MACE_KERNELS_TRANSFORMER_H_
#define MACE_KERNELS_TRANSFORMER_H_

#include "mace/core/transformer.h"
#include "mace/ops/opencl/common.h"

namespace mace {
class OpContext;
namespace ops {

class Transformer : public TransformerBase {
 public:
  // Transform source tensor to target.
  std::vector<std::unique_ptr<OperatorDef>> ConstructTranformOp(
      OperatorDef *op_def,
      bool transform_filter = true) override;
 private:
  std::unique_ptr<OperatorDef> DoTransform(
      mace::OperatorDef *op_def,
      const int input_idx,
      const mace::DataType dt,
      const BufferType buffer_type,
      const MemoryType mem_type);
};


}  // namespace ops
}  // namespace mace

#endif  // MACE_KERNELS_TENSOR_TRANSFORMER_H_
