// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/opencl/buffer_transformer.h"

namespace mace {
namespace ops {

std::string TransformedFilterName(const std::string &name) {
  // TODO(liuqi): This may create a conflict.
  const char *postfix = "_mace_identity_transformed";
  return name + postfix;
}

MaceStatus TransformFilter(
    mace::OpConstructContext *context,
    OperatorDef *op_def,
    const int input_idx,
    const OpenCLBufferType buffer_type,
    const MemoryType mem_type,
    const int wino_blk_size) {
  OpContext op_context(context->workspace(), context->device());
  Workspace *ws = context->workspace();
  std::string input_name = op_def->input(input_idx);
  Tensor *input = ws->GetTensor(input_name);
  const DataType dt = input->dtype();
  std::string output_name = TransformedFilterName(input_name);
  Tensor *output =
      ws->CreateTensor(output_name, context->device()->allocator(), dt, true);

  // update the information
  op_def->set_input(input_idx, output_name);
  input->MarkUnused();
  return OpenCLBufferTransformer(input->memory_type(), mem_type).
      Transform(&op_context, input, buffer_type, mem_type, wino_blk_size,
                output);
}

}  // namespace ops
}  // namespace mace
