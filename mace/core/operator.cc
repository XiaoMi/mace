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

#include <sstream>
#include <map>
#include <memory>
#include <vector>

#include "mace/core/operator.h"

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

OpConstructContext::OpConstructContext(Workspace *ws)
    : operator_def_(nullptr),
      ws_(ws),
      device_(nullptr) {}

void OpConstructContext::set_operator_def(
    std::shared_ptr<OperatorDef> operator_def) {
  operator_def_ = operator_def;
}

OpInitContext::OpInitContext(Workspace *ws, Device *device)
    : ws_(ws), device_(device) {}

Operation::Operation(OpConstructContext *context)
    : operator_def_(context->operator_def())
{}

MaceStatus Operation::Init(OpInitContext *context) {
  Workspace *ws = context->workspace();
  for (const std::string &input_str : operator_def_->input()) {
    const Tensor *tensor = ws->GetTensor(input_str);
    MACE_CHECK(tensor != nullptr, "op ", operator_def_->type(),
               ": Encountered a non-existing input tensor: ", input_str);
    inputs_.push_back(tensor);
  }
  for (int i = 0; i < operator_def_->output_size(); ++i) {
    const std::string output_str = operator_def_->output(i);
    if (ws->HasTensor(output_str)) {
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
    }
    if (i < operator_def_->output_shape_size()) {
      std::vector<index_t>
          shape_configured(operator_def_->output_shape(i).dims_size());
      for (size_t dim = 0; dim < shape_configured.size(); ++dim) {
        shape_configured[dim] = operator_def_->output_shape(i).dims(dim);
      }
      ws->GetTensor(output_str)->SetShapeConfigured(shape_configured);
    }
  }
  return MaceStatus::MACE_SUCCESS;
}

// op registry
namespace {
class OpKeyBuilder {
 public:
  explicit OpKeyBuilder(const std::string &op_name);

  OpKeyBuilder &Device(DeviceType device);

  OpKeyBuilder &TypeConstraint(const char *attr_name,
                               DataType allowed);

  const std::string Build();

 private:
  std::string op_name_;
  DeviceType device_type_;
  std::map<std::string, DataType> type_constraint_;
};

OpKeyBuilder::OpKeyBuilder(const std::string &op_name) : op_name_(op_name) {}

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
}  // namespace

OpRegistrationInfo::OpRegistrationInfo() {
  // default device type placer
  device_placer = [this](OpConditionContext *context) -> std::set<DeviceType> {
    MACE_UNUSED(context);
    return this->devices;
  };

  // default input and output memory type setter
  memory_type_setter = [](OpConditionContext *context) -> void {
    if (context->device()->device_type() == DeviceType::GPU) {
#ifdef MACE_ENABLE_OPENCL
      if (context->device()->gpu_runtime()->UseImageMemory()) {
        context->set_output_mem_type(MemoryType::GPU_IMAGE);
      } else {
        context->set_output_mem_type(MemoryType::GPU_BUFFER);
      }
#endif  // MACE_ENABLE_OPENCL
    } else {
      context->set_output_mem_type(MemoryType::CPU_BUFFER);
    }
  };

  data_format_selector = [](OpConditionContext *context)
      -> std::vector<DataFormat> {
    DataFormat op_data_format =
        static_cast<DataFormat>(
            ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                *context->operator_def(), "data_format",
                static_cast<int>(DataFormat::NONE)));
    return std::vector<DataFormat>(context->operator_def()->input_size(),
                                   op_data_format);
  };
}

void OpRegistrationInfo::AddDevice(DeviceType device) {
  devices.insert(device);
}

void OpRegistrationInfo::Register(const std::string &key, OpCreator creator) {
  VLOG(3) << "Registering: " << key;
  MACE_CHECK(creators.count(key) == 0, "Key already registered: ", key);
  creators[key] = creator;
}

MaceStatus OpRegistryBase::Register(
    const std::string &op_type,
    const DeviceType device_type,
    const DataType dt,
    OpRegistrationInfo::OpCreator creator) {
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<OpRegistrationInfo>(
        new OpRegistrationInfo);
  }
  registry_[op_type]->AddDevice(device_type);

  std::string op_key = OpKeyBuilder(op_type)
      .Device(device_type)
      .TypeConstraint("T", dt)
      .Build();
  registry_.at(op_type)->Register(op_key, creator);
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus OpRegistryBase::Register(
    const OpConditionBuilder &builder) {
  std::string op_type = builder.type();
  if (registry_.count(op_type) == 0) {
    registry_[op_type] = std::unique_ptr<OpRegistrationInfo>(
        new OpRegistrationInfo);
  }
  builder.Finalize(registry_[op_type].get());
  return MaceStatus::MACE_SUCCESS;
}

const std::set<DeviceType> OpRegistryBase::AvailableDevices(
    const std::string &op_type, OpConditionContext *context) const {
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");

  return registry_.at(op_type)->device_placer(context);
}

void OpRegistryBase::GetInOutMemoryTypes(
    const std::string &op_type,
    OpConditionContext *context) const {
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");
  return registry_.at(op_type)->memory_type_setter(context);
}

const std::vector<DataFormat> OpRegistryBase::InputsDataFormat(
    const std::string &op_type,
    OpConditionContext *context) const {
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");
  return registry_.at(op_type)->data_format_selector(context);
}

std::unique_ptr<Operation> OpRegistryBase::CreateOperation(
    OpConstructContext *context,
    DeviceType device_type) const {
  auto operator_def = context->operator_def();
  DataType dtype = static_cast<DataType>(
      ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
          *operator_def, "T", static_cast<int>(DT_FLOAT)));
  VLOG(1) << "Creating operator " << operator_def->name() << "("
          << operator_def->type() << "<" << dtype << ">" << ") on "
          << device_type;
  const std::string op_type = context->operator_def()->type();
  MACE_CHECK(registry_.count(op_type) != 0,
             op_type, " operation is not registered.");

  std::string key = OpKeyBuilder(op_type)
      .Device(device_type)
      .TypeConstraint("T", dtype)
      .Build();
  if (registry_.at(op_type)->creators.count(key) == 0) {
    LOG(FATAL) << "Key not registered: " << key;
  }
  return registry_.at(op_type)->creators.at(key)(context);
}

OpConditionBuilder::OpConditionBuilder(const std::string &type)
  : type_(type) {}

const std::string OpConditionBuilder::type() const {
  return type_;
}

OpConditionBuilder &OpConditionBuilder::SetDevicePlacerFunc(
    OpRegistrationInfo::DevicePlacer placer) {
  placer_ = placer;
  return *this;
}

OpConditionBuilder& OpConditionBuilder::SetInputMemoryTypeSetter(
    OpRegistrationInfo::MemoryTypeSetter setter) {
  memory_type_setter_ = setter;
  return *this;
}

OpConditionBuilder& OpConditionBuilder::SetInputsDataFormatSelector(
    OpRegistrationInfo::DataFormatSelector selector) {
  data_format_selector_ = selector;
  return *this;
}

void OpConditionBuilder::Finalize(OpRegistrationInfo *info) const {
  if (info != nullptr) {
    if (placer_) {
      info->device_placer = placer_;
    }
    if (memory_type_setter_) {
      info->memory_type_setter = memory_type_setter_;
    }

    if (data_format_selector_) {
      info->data_format_selector = data_format_selector_;
    }
  }
}

}  // namespace mace
