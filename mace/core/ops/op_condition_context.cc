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

#include "mace/core/ops/op_condition_context.h"

#include "mace/core/arg_helper.h"
#include "mace/proto/mace.pb.h"
#include "mace/utils/logging.h"

namespace mace {

OpConditionContext::OpConditionContext(
    const Workspace *ws,
    OpConditionContext::TensorShapeMap *info)
    : operator_def_(nullptr),
      ws_(ws),
      device_(nullptr),
      tensor_shape_info_(info) {}

void OpConditionContext::set_operator_def(
    const OperatorDef *operator_def) {
  operator_def_ = operator_def;
  input_data_types_.clear();
}

void OpConditionContext::SetInputInfo(size_t idx,
                                      MemoryType mem_type,
                                      DataType dt) {
  if (input_mem_types_.empty()) {
    // the default inputs' memory types are same as output memory type.
    input_mem_types_.resize(operator_def_->input_size(), output_mem_type_);
  }
  if (input_data_types_.empty()) {
    // the default inputs' data types are same as operation's data type.
    DataType op_dt = static_cast<DataType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            *operator_def_, "T", static_cast<int>(DataType::DT_FLOAT)));
    input_data_types_.resize(operator_def_->input_size(), op_dt);
  }
  MACE_CHECK(idx < input_mem_types_.size() && idx < input_data_types_.size());
  input_mem_types_[idx] = mem_type;
  input_data_types_[idx] = dt;
}

void OpConditionContext::set_output_mem_type(MemoryType type) {
  MACE_CHECK(operator_def_ != nullptr);
  output_mem_type_ = type;
  input_mem_types_.clear();
}

MemoryType OpConditionContext::GetInputMemType(size_t idx) const {
  if (input_mem_types_.empty()) {
    return output_mem_type_;
  }
  MACE_CHECK(idx < input_mem_types_.size(),
             idx, " < ", input_mem_types_.size());
  return input_mem_types_[idx];
}

DataType OpConditionContext::GetInputDataType(size_t idx) const {
  if (input_data_types_.empty()) {
    // the default inputs' data types are same as operation's data type.
    return static_cast<DataType>(
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            *operator_def_, "T", static_cast<int>(DataType::DT_FLOAT)));
  }
  MACE_CHECK(idx < input_data_types_.size());
  return input_data_types_[idx];
}

#ifdef MACE_ENABLE_OPENCL
void OpConditionContext::SetInputOpenCLBufferType(
    size_t idx, OpenCLBufferType buffer_type) {
  if (input_opencl_buffer_types_.empty()) {
    // the default inputs' memory types are same as output memory type.
    input_opencl_buffer_types_.resize(operator_def_->input_size(),
                                      OpenCLBufferType::IN_OUT_CHANNEL);
  }
  MACE_CHECK(idx < input_opencl_buffer_types_.size());
  input_opencl_buffer_types_[idx] = buffer_type;
}

OpenCLBufferType OpConditionContext::GetInputOpenCLBufferType(
    size_t idx) const {
  if (input_opencl_buffer_types_.empty()) {
    return OpenCLBufferType::IN_OUT_CHANNEL;
  }
  MACE_CHECK(idx < input_opencl_buffer_types_.size());
  return input_opencl_buffer_types_[idx];
}
#endif  // MACE_ENABLE_OPENCL

}  // namespace mace
