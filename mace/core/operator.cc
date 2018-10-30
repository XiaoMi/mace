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

#include <sstream>
#include <memory>
#include <vector>

#include "mace/core/operator.h"

namespace mace {

OpConstructContext::OpConstructContext(Workspace *ws)
    : operator_def_(nullptr), ws_(ws), device_(nullptr) {}
OpConstructContext::OpConstructContext(OperatorDef *operator_def,
                                       Workspace *ws,
                                       Device *device)
    : operator_def_(operator_def), ws_(ws), device_(device) {}

OpInitContext::OpInitContext(Workspace *ws, Device *device)
    : ws_(ws), device_(device) {}

Operation::Operation(OpConstructContext *context)
    : operator_def_(std::make_shared<OperatorDef>(*(context->operator_def())))
{}

MaceStatus Operation::Init(OpInitContext *context) {
  Workspace *ws = context->workspace();
  for (const std::string &input_str : operator_def_->input()) {
    const Tensor *tensor = ws->GetTensor(input_str);
    MACE_CHECK(tensor != nullptr, "op ", operator_def_->type(),
               ": Encountered a non-existing input tensor: ", input_str);
    inputs_.push_back(tensor);
  }
  // TODO(liuqi): filter transform
  for (int i = 0; i < operator_def_->output_size(); ++i) {
    const std::string output_str = operator_def_->output(i);
    if (ws->HasTensor(output_str)) {
      // TODO(liuqi): Workspace should pre-allocate all of the output tensors
      outputs_.push_back(ws->GetTensor(output_str));
    } else {
      MACE_CHECK(
          operator_def_->output_type_size() == 0 ||
              operator_def_->output_size() == operator_def_->output_type_size(),
          "operator output size != operator output type size",
          operator_def_->output_size(),
          operator_def_->output_type_size());
      DataType output_type;
      if (i < operator_def_->output_type_size()) {
        output_type = operator_def_->output_type(i);
      } else {
        output_type = static_cast<DataType>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            *operator_def_, "T", static_cast<int>(DT_FLOAT)));
      }
      outputs_.push_back(MACE_CHECK_NOTNULL(ws->CreateTensor(
          output_str, context->device()->allocator(), output_type)));

      if (i < operator_def_->output_shape_size()) {
        std::vector<index_t>
            shape_configured(operator_def_->output_shape(i).dims_size());
        for (size_t dim = 0; dim < shape_configured.size(); ++dim) {
          shape_configured[dim] = operator_def_->output_shape(i).dims(dim);
        }
        ws->GetTensor(output_str)->SetShapeConfigured(shape_configured);
      }
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

OpKeyBuilder::OpKeyBuilder(const char *op_name) : op_name_(op_name) {}

OpKeyBuilder &OpKeyBuilder::Device(DeviceType device) {
  device_type_ = device;
  return *this;
}

OpKeyBuilder &OpKeyBuilder::TypeConstraint(const char *attr_name,
                                           DataType allowed) {
  type_constraint_[attr_name] = allowed;
  return *this;
}

const std::string OpKeyBuilder::Build() {
  static const std::vector<std::string> type_order = {"T"};
  std::stringstream ss;
  ss << op_name_;
  ss << device_type_;
  for (auto type : type_order) {
    ss << type << "_" << DataTypeToString(type_constraint_[type]);
  }

  return ss.str();
}

OpRegistryBase::~OpRegistryBase() = default;

std::unique_ptr<Operation> OpRegistryBase::CreateOperation(
    OpConstructContext *context,
    DeviceType device_type,
    const NetMode mode) const {
  OperatorDef *operator_def = context->operator_def();
  const int dtype = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
      *operator_def, "T", static_cast<int>(DT_FLOAT));
  const int op_mode_i = ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
      *operator_def, "mode", static_cast<int>(NetMode::NORMAL));
  const NetMode op_mode = static_cast<NetMode>(op_mode_i);
  VLOG(3) << "Creating operator " << operator_def->name() << "("
          << operator_def->type() << "<" << dtype << ">" << ") on "
          << device_type;
  if (op_mode == mode) {
    return registry_.Create(
        OpKeyBuilder(operator_def->type().data())
            .Device(device_type)
            .TypeConstraint("T", static_cast<DataType>(dtype))
            .Build(),
        context);
  } else {
    return nullptr;
  }
}

}  // namespace mace
