//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_OPERATOR_H
#define MACE_CORE_OPERATOR_H

#include "mace/core/common.h"
#include "mace/core/proto_utils.h"
#include "mace/core/registry.h"
#include "mace/core/tensor.h"
#include "mace/core/workspace.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class OperatorBase {
 public:
  explicit OperatorBase(const OperatorDef &operator_def, Workspace *ws);
  virtual ~OperatorBase() noexcept {}

  inline bool HasArgument(const string &name) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasArgument(*operator_def_, name);
  }
  template <typename T>
  inline T GetSingleArgument(const string &name, const T &default_value) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetSingleArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }
  template <typename T>
  inline bool HasSingleArgumentOfType(const string &name) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::HasSingleArgumentOfType<OperatorDef, T>(
        *operator_def_, name);
  }
  template <typename T>
  inline vector<T> GetRepeatedArgument(
      const string &name, const vector<T> &default_value = {}) const {
    MACE_CHECK(operator_def_, "operator_def was null!");
    return ArgumentHelper::GetRepeatedArgument<OperatorDef, T>(
        *operator_def_, name, default_value);
  }

  inline const Tensor *Input(unsigned int idx) {
    MACE_CHECK(idx < inputs_.size());
    return inputs_[idx];
  }

  inline Tensor *Output(int idx) { return outputs_[idx]; }

  inline int InputSize() { return inputs_.size(); }
  inline int OutputSize() { return outputs_.size(); }
  inline const vector<const Tensor *> &Inputs() const { return inputs_; }
  inline const vector<Tensor *> &Outputs() { return outputs_; }

  virtual bool Run() = 0;

  inline const OperatorDef &debug_def() const {
    MACE_CHECK(has_debug_def(), "operator_def was null!");
    return *operator_def_;
  }

  inline void set_debug_def(
      const std::shared_ptr<const OperatorDef> &operator_def) {
    operator_def_ = operator_def;
  }

  inline bool has_debug_def() const { return operator_def_ != nullptr; }

 protected:
  Workspace *operator_ws_;
  std::shared_ptr<const OperatorDef> operator_def_;
  vector<const Tensor *> inputs_;
  vector<Tensor *> outputs_;

  DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

template <DeviceType D, class T>
class Operator : public OperatorBase {
 public:
  explicit Operator(const OperatorDef &operator_def, Workspace *ws)
      : OperatorBase(operator_def, ws) {
    for (const string &input_str : operator_def.input()) {
      const Tensor *tensor = ws->GetTensor(input_str);
      MACE_CHECK(tensor != nullptr, "op ", operator_def.type(),
                 ": Encountered a non-existing input tensor: ", input_str);
      inputs_.push_back(tensor);
    }

    for (const string &output_str : operator_def.output()) {
      outputs_.push_back(MACE_CHECK_NOTNULL(ws->CreateTensor(
          output_str, GetDeviceAllocator(D), DataTypeToEnum<T>::v())));
    }
  }
  virtual bool Run() override = 0;
  ~Operator() noexcept override {}
};

// OP_INPUT_TAGS and OP_OUTPUT_TAGS are optional features to name the indices of
// the
// operator's inputs and outputs, in order to avoid confusion. For example, for
// a fully convolution layer that has input, weight and bias, you can define its
// input tags as:
//     OP_INPUT_TAGS(INPUT, WEIGHT, BIAS);
// And in the code, instead of doing
//     auto& weight = Input(1);
// you can now do
//     auto& weight = Input(WEIGHT);
// to make it more clear.
#define OP_INPUT_TAGS(first_input, ...) \
  enum _InputTags { first_input = 0, __VA_ARGS__ }
#define OP_OUTPUT_TAGS(first_input, ...) \
  enum _OutputTags { first_input = 0, __VA_ARGS__ }

typedef Registry<std::string, OperatorBase, const OperatorDef &, Workspace *>
    OperatorRegistry;
typedef Registry<std::string, OperatorBase, const OperatorDef &, Workspace *> *(
    *RegistryFunction)();
std::map<int32_t, OperatorRegistry *> *gDeviceTypeRegistry();

struct DeviceTypeRegisterer {
  explicit DeviceTypeRegisterer(int32_t type, RegistryFunction func) {
    if (gDeviceTypeRegistry()->count(type)) {
      LOG(ERROR) << "Device type " << type
                 << "registered twice. This should not happen. Did you have "
                    "duplicated numbers assigned to different devices?";
      std::exit(1);
    }
    // Calling the registry function to get the actual registry pointer.
    gDeviceTypeRegistry()->emplace(type, func());
  }
};

#define MACE_REGISTER_DEVICE_TYPE(type, registry_function)         \
  namespace {                                                      \
  static DeviceTypeRegisterer MACE_ANONYMOUS_VARIABLE(DeviceType)( \
      type, &registry_function);                                   \
  }

MACE_DECLARE_REGISTRY(CPUOperatorRegistry,
                      OperatorBase,
                      const OperatorDef &,
                      Workspace *);

#define REGISTER_CPU_OPERATOR_CREATOR(key, ...) \
  MACE_REGISTER_CREATOR(CPUOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_CPU_OPERATOR(name, ...) \
  MACE_REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

MACE_DECLARE_REGISTRY(NEONOperatorRegistry,
                      OperatorBase,
                      const OperatorDef &,
                      Workspace *);

#define REGISTER_NEON_OPERATOR_CREATOR(key, ...) \
  MACE_REGISTER_CREATOR(NEONOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_NEON_OPERATOR(name, ...) \
  MACE_REGISTER_CLASS(NEONOperatorRegistry, name, __VA_ARGS__)

MACE_DECLARE_REGISTRY(OPENCLOperatorRegistry,
                      OperatorBase,
                      const OperatorDef &,
                      Workspace *);

#define REGISTER_OPENCL_OPERATOR_CREATOR(key, ...) \
  MACE_REGISTER_CREATOR(OPENCLOperatorRegistry, key, __VA_ARGS__)
#define REGISTER_OPENCL_OPERATOR(name, ...) \
  MACE_REGISTER_CLASS(OPENCLOperatorRegistry, name, __VA_ARGS__)

unique_ptr<OperatorBase> CreateOperator(const OperatorDef &operator_def,
                                        Workspace *ws,
                                        DeviceType type);

}  //  namespace mace

#endif  // MACE_CORE_OPERATOR_H
