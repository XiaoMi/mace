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

#include "mace/ops/transformer.h"

#include <string>
#include <memory>

namespace mace {
namespace ops {

std::unique_ptr<OperatorDef> Transformer::DoTransform(
    mace::OperatorDef *op_def,
    const int input_idx,
    const mace::DataType dt,
    const BufferType buffer_type,
    const MemoryType mem_type) {
  int32_t device = op_def->device_type();
  std::string input_name = op_def->input(input_idx);
  std::string output_name = input_name + "_transformed";

  op_def->set_input(input_idx, output_name);
  std::unique_ptr<OperatorDef> op(new OperatorDef);
  op->set_name(output_name);
  op->set_type("BufferTransform");
  op->add_input(input_name);
  op->add_output(output_name);
  Argument *arg = op->add_arg();
  arg->set_name("buffer_type");
  arg->set_i(static_cast<int32_t>(buffer_type));
  arg = op->add_arg();
  arg->set_name("mem_type");
  arg->set_i(static_cast<int32_t>(mem_type));
  arg = op->add_arg();
  arg->set_name("T");
  arg->set_i(static_cast<int32_t>(dt));
  arg = op->add_arg();
  arg->set_name("device");
  arg->set_i(device);

  return std::move(op);
}

}  // namespace ops
}  // namespace mace
