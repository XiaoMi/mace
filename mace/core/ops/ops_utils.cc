// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#include "mace/core/ops/ops_utils.h"

#include "mace/core/proto/arg_helper.h"

namespace mace {

void OpsUtils::BuildTransformOpDef(
    const std::string &input_name,
    const std::vector<index_t> &input_shape,
    const std::string &output_name,
    const DataType dt,
    const BufferContentType content_type,
    const MemoryType mem_type,
    DataFormat data_format,
    OperatorDef *op_def) {
  std::string op_name = "mace_node_" + output_name;
  op_def->set_name(op_name);
  op_def->set_type("BufferTransform");
  op_def->add_input(input_name);
  op_def->add_output(output_name);
  op_def->set_device_type(static_cast<DeviceType>(RT_OPENCL));
  Argument *arg = op_def->add_arg();
  arg->set_name("content_type");
  arg->set_i(static_cast<int32_t>(content_type));
  arg = op_def->add_arg();
  arg->set_name(OutputMemoryTypeTagName());
  arg->set_i(static_cast<int32_t>(mem_type));
  arg = op_def->add_arg();
  arg->set_name("T");
  arg->set_i(static_cast<int32_t>(dt));
  arg = op_def->add_arg();
  arg->set_name("data_format");
  arg->set_i(static_cast<int>(data_format));
  if (!input_shape.empty()) {
    OutputShape *shape = op_def->add_output_shape();
    for (auto value : input_shape) {
      shape->add_dims(value);
    }
  }
}

}  // namespace mace
