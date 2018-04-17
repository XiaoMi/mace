//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <memory>
#include <numeric>

#include "mace/public/mace_types.h"
#include "mace/utils/logging.h"

namespace mace {

ConstTensor::ConstTensor(const std::string &name,
                         const unsigned char *data,
                         const std::vector<int64_t> &dims,
                         const DataType data_type,
                         uint32_t node_id)
    : name_(name),
      data_(data),
      data_size_(std::accumulate(
          dims.begin(), dims.end(), 1, std::multiplies<int64_t>())),
      dims_(dims.begin(), dims.end()),
      data_type_(data_type),
      node_id_(node_id) {}

ConstTensor::ConstTensor(const std::string &name,
                         const unsigned char *data,
                         const std::vector<int64_t> &dims,
                         const int data_type,
                         uint32_t node_id)
    : name_(name),
      data_(data),
      data_size_(std::accumulate(
          dims.begin(), dims.end(), 1, std::multiplies<int64_t>())),
      dims_(dims.begin(), dims.end()),
      data_type_(static_cast<DataType>(data_type)),
      node_id_(node_id) {}

const std::string &ConstTensor::name() const { return name_; }
const unsigned char *ConstTensor::data() const { return data_; }
int64_t ConstTensor::data_size() const { return data_size_; }
const std::vector<int64_t> &ConstTensor::dims() const { return dims_; }
DataType ConstTensor::data_type() const { return data_type_; }
uint32_t ConstTensor::node_id() const { return node_id_; }

Argument::Argument() : has_bits_(0) {}

void Argument::CopyFrom(const Argument &from) {
  this->name_ = from.name();
  this->f_ = from.f();
  this->i_ = from.i();
  this->s_ = from.s();
  auto floats = from.floats();
  this->floats_.resize(floats.size());
  std::copy(floats.begin(), floats.end(), this->floats_.begin());
  auto ints = from.ints();
  this->ints_.resize(ints.size());
  std::copy(ints.begin(), ints.end(), this->ints_.begin());
  auto strings = from.floats();
  this->strings_.resize(strings.size());
  std::copy(floats.begin(), floats.end(), this->floats_.begin());

  this->has_bits_ = from.has_bits_;
}
const std::string &Argument::name() const { return name_; }
void Argument::set_name(const std::string &value) { name_ = value; }
bool Argument::has_f() const { return (has_bits_ & 0x00000001u) != 0; }
void Argument::set_has_f() { has_bits_ |= 0x00000001u; }
float Argument::f() const { return f_; }
void Argument::set_f(float value) {
  set_has_f();
  f_ = value;
}
bool Argument::has_i() const { return (has_bits_ & 0x00000002u) != 0; }
void Argument::set_has_i() { has_bits_ |= 0x00000002u; }
int64_t Argument::i() const { return i_; }
void Argument::set_i(int64_t value) {
  set_has_i();
  i_ = value;
}
bool Argument::has_s() const { return (has_bits_ & 0x00000004u) != 0; }
void Argument::set_has_s() { has_bits_ |= 0x00000004u; }
std::string Argument::s() const { return s_; }
void Argument::set_s(const std::string &value) {
  set_has_s();
  s_ = value;
}
const std::vector<float> &Argument::floats() const { return floats_; }
void Argument::add_floats(float value) { floats_.push_back(value); }
void Argument::set_floats(const std::vector<float> &value) {
  floats_.resize(value.size());
  std::copy(value.begin(), value.end(), floats_.begin());
}
const std::vector<int64_t> &Argument::ints() const { return ints_; }
void Argument::add_ints(int64_t value) { ints_.push_back(value); }
void Argument::set_ints(const std::vector<int64_t> &value) {
  ints_.resize(value.size());
  std::copy(value.begin(), value.end(), ints_.begin());
}
const std::vector<std::string> &Argument::strings() const { return strings_; }
void Argument::add_strings(const ::std::string &value) {
  strings_.push_back(value);
}
void Argument::set_strings(const std::vector<std::string> &value) {
  strings_.resize(value.size());
  std::copy(value.begin(), value.end(), strings_.begin());
}

// Node Input
NodeInput::NodeInput(int node_id, int output_port)
    : node_id_(node_id), output_port_(output_port) {}
void NodeInput::CopyFrom(const NodeInput &from) {
  node_id_ = from.node_id();
  output_port_ = from.output_port();
}
int NodeInput::node_id() const { return node_id_; }
void NodeInput::set_node_id(int node_id) { node_id_ = node_id; }
int NodeInput::output_port() const { return output_port_; }
void NodeInput::set_output_port(int output_port) { output_port_ = output_port; }

// OutputShape
OutputShape::OutputShape() {}
OutputShape::OutputShape(const std::vector<int64_t> &dims)
    : dims_(dims.begin(), dims.end()) {}
void OutputShape::CopyFrom(const OutputShape &from) {
  auto from_dims = from.dims();
  dims_.resize(from_dims.size());
  std::copy(from_dims.begin(), from_dims.end(), dims_.begin());
}
const std::vector<int64_t> &OutputShape::dims() const { return dims_; }

// Operator Def
void OperatorDef::CopyFrom(const OperatorDef &from) {
  name_ = from.name();
  type_ = from.type();

  auto from_input = from.input();
  input_.resize(from_input.size());
  std::copy(from_input.begin(), from_input.end(), input_.begin());
  auto from_output = from.output();
  output_.resize(from_output.size());
  std::copy(from_output.begin(), from_output.end(), output_.begin());
  auto from_arg = from.arg();
  arg_.resize(from_arg.size());
  for (size_t i = 0; i < from_arg.size(); ++i) {
    arg_[i].CopyFrom(from_arg[i]);
  }
  auto from_output_shape = from.output_shape();
  output_shape_.resize(from_output_shape.size());
  for (size_t i = 0; i < from_output_shape.size(); ++i) {
    output_shape_[i].CopyFrom(from_output_shape[i]);
  }
  auto from_data_type = from.output_type();
  output_type_.resize(from_data_type.size());
  std::copy(from_data_type.begin(), from_data_type.end(), output_type_.begin());

  auto mem_ids = from.mem_id();
  mem_id_.resize(mem_ids.size());
  std::copy(mem_ids.begin(), mem_ids.end(), mem_id_.begin());

  // nnlib
  node_id_ = from.node_id();
  op_id_ = from.op_id();
  padding_ = from.padding();
  auto from_node_input = from.node_input();
  node_input_.resize(from_node_input.size());
  for (size_t i = 0; i < from_node_input.size(); ++i) {
    node_input_[i].CopyFrom(from_node_input[i]);
  }
  auto from_out_max_byte_size = from.out_max_byte_size();
  out_max_byte_size_.resize(from_out_max_byte_size.size());
  std::copy(from_out_max_byte_size.begin(), from_out_max_byte_size.end(),
            out_max_byte_size_.begin());

  has_bits_ = from.has_bits_;
}

const std::string &OperatorDef::name() const { return name_; }
void OperatorDef::set_name(const std::string &name_) {
  set_has_name();
  OperatorDef::name_ = name_;
}
bool OperatorDef::has_name() const { return (has_bits_ & 0x00000001u) != 0; }
void OperatorDef::set_has_name() { has_bits_ |= 0x00000001u; }
const std::string &OperatorDef::type() const { return type_; }
void OperatorDef::set_type(const std::string &type_) {
  set_has_type();
  OperatorDef::type_ = type_;
}
bool OperatorDef::has_type() const { return (has_bits_ & 0x00000002u) != 0; }
void OperatorDef::set_has_type() { has_bits_ |= 0x00000002u; }
const std::vector<int> &OperatorDef::mem_id() const { return mem_id_; }
void OperatorDef::set_mem_id(const std::vector<int> &value) {
  mem_id_.resize(value.size());
  std::copy(value.begin(), value.end(), mem_id_.begin());
}
uint32_t OperatorDef::node_id() const { return node_id_; }
void OperatorDef::set_node_id(uint32_t node_id) { node_id_ = node_id; }
uint32_t OperatorDef::op_id() const { return op_id_; }
uint32_t OperatorDef::padding() const { return padding_; }
void OperatorDef::set_padding(uint32_t padding) { padding_ = padding; }
const std::vector<NodeInput> &OperatorDef::node_input() const {
  return node_input_;
}
void OperatorDef::add_node_input(const NodeInput &value) {
  node_input_.push_back(value);
}
const std::vector<int> &OperatorDef::out_max_byte_size() const {
  return out_max_byte_size_;
}
void OperatorDef::add_out_max_byte_size(int value) {
  out_max_byte_size_.push_back(value);
}
const std::vector<std::string> &OperatorDef::input() const { return input_; }
const std::string &OperatorDef::input(int index) const {
  MACE_CHECK(0 <= index && index <= static_cast<int>(input_.size()));
  return input_[index];
}
std::string *OperatorDef::add_input() {
  input_.push_back("");
  return &input_.back();
}
void OperatorDef::add_input(const ::std::string &value) {
  input_.push_back(value);
}
void OperatorDef::add_input(::std::string &&value) { input_.push_back(value); }
void OperatorDef::set_input(const std::vector<std::string> &value) {
  input_.resize(value.size());
  std::copy(value.begin(), value.end(), input_.begin());
}
const std::vector<std::string> &OperatorDef::output() const { return output_; }
const std::string &OperatorDef::output(int index) const {
  MACE_CHECK(0 <= index && index <= static_cast<int>(output_.size()));
  return output_[index];
}
std::string *OperatorDef::add_output() {
  output_.push_back("");
  return &output_.back();
}
void OperatorDef::add_output(const ::std::string &value) {
  output_.push_back(value);
}
void OperatorDef::add_output(::std::string &&value) {
  output_.push_back(value);
}
void OperatorDef::set_output(const std::vector<std::string> &value) {
  output_.resize(value.size());
  std::copy(value.begin(), value.end(), output_.begin());
}
const std::vector<Argument> &OperatorDef::arg() const { return arg_; }
Argument *OperatorDef::add_arg() {
  arg_.emplace_back(Argument());
  return &arg_.back();
}
const std::vector<OutputShape> &OperatorDef::output_shape() const {
  return output_shape_;
}
void OperatorDef::add_output_shape(const OutputShape &value) {
  output_shape_.push_back(value);
}
const std::vector<DataType> &OperatorDef::output_type() const {
  return output_type_;
}
void OperatorDef::set_output_type(const std::vector<DataType> &value) {
  output_type_.resize(value.size());
  std::copy(value.begin(), value.end(), output_type_.begin());
}

// MemoryBlock
MemoryBlock::MemoryBlock(int mem_id, uint32_t x, uint32_t y)
    : mem_id_(mem_id), x_(x), y_(y) {}

int MemoryBlock::mem_id() const { return mem_id_; }
uint32_t MemoryBlock::x() const { return x_; }
uint32_t MemoryBlock::y() const { return y_; }

// MemoryArena
const std::vector<MemoryBlock> &MemoryArena::mem_block() const {
  return mem_block_;
}
std::vector<MemoryBlock> &MemoryArena::mutable_mem_block() {
  return mem_block_;
}
int MemoryArena::mem_block_size() const { return mem_block_.size(); }

// InputInfo
const std::string &InputInfo::name() const { return name_; }
int32_t InputInfo::node_id() const { return node_id_; }
int32_t InputInfo::max_byte_size() const { return max_byte_size_; }
DataType InputInfo::data_type() const { return data_type_; }
const std::vector<int32_t> &InputInfo::dims() const { return dims_; }

// OutputInfo
const std::string &OutputInfo::name() const { return name_; }
int32_t OutputInfo::node_id() const { return node_id_; }
int32_t OutputInfo::max_byte_size() const { return max_byte_size_; }
DataType OutputInfo::data_type() const { return data_type_; }
void OutputInfo::set_data_type(DataType data_type) { data_type_ = data_type; }
const std::vector<int32_t> &OutputInfo::dims() const { return dims_; }
void OutputInfo::set_dims(const std::vector<int32_t> &dims) { dims_ = dims; }

// NetDef
NetDef::NetDef() : has_bits_(0) {}

const std::string &NetDef::name() const { return name_; }
void NetDef::set_name(const std::string &value) {
  set_has_name();
  name_ = value;
}
bool NetDef::has_name() const { return (has_bits_ & 0x00000001u) != 0; }
void NetDef::set_has_name() { has_bits_ |= 0x00000001u; }
const std::string &NetDef::version() const { return version_; }
void NetDef::set_version(const std::string &value) {
  set_has_version();
  version_ = value;
}
bool NetDef::has_version() const { return (has_bits_ & 0x00000002u) != 0; }
void NetDef::set_has_version() { has_bits_ |= 0x00000002u; }
const std::vector<OperatorDef> &NetDef::op() const { return op_; }
OperatorDef *NetDef::add_op() {
  op_.emplace_back(OperatorDef());
  return &op_.back();
}
std::vector<OperatorDef> &NetDef::mutable_op() { return op_; }
const std::vector<Argument> &NetDef::arg() const { return arg_; }
Argument *NetDef::add_arg() {
  arg_.emplace_back(Argument());
  return &arg_.back();
}
std::vector<Argument> &NetDef::mutable_arg() { return arg_; }
const std::vector<ConstTensor> &NetDef::tensors() const { return tensors_; }
std::vector<ConstTensor> &NetDef::mutable_tensors() { return tensors_; }
const MemoryArena &NetDef::mem_arena() const { return mem_arena_; }
MemoryArena &NetDef::mutable_mem_arena() {
  set_has_mem_arena();
  return mem_arena_;
}
bool NetDef::has_mem_arena() const { return (has_bits_ & 0x00000004u) != 0; }
void NetDef::set_has_mem_arena() { has_bits_ |= 0x00000004u; }
const std::vector<InputInfo> &NetDef::input_info() const { return input_info_; }
const std::vector<OutputInfo> &NetDef::output_info() const {
  return output_info_;
}
std::vector<OutputInfo> &NetDef::mutable_output_info() { return output_info_; }

int NetDef::op_size() const { return op_.size(); }

const OperatorDef &NetDef::op(const int idx) const {
  MACE_CHECK(0 <= idx && idx < op_size());
  return op_[idx];
}

};  // namespace mace
