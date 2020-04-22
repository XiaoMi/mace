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


#ifndef MACE_CORE_REGISTRY_OPS_REGISTRY_H_
#define MACE_CORE_REGISTRY_OPS_REGISTRY_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "mace/core/bfloat16.h"
#include "mace/core/types.h"
#include "mace/core/ops/operator.h"
#include "mace/core/ops/op_condition_builder.h"
#include "mace/core/ops/op_condition_context.h"
#include "mace/public/mace.h"
#include "mace/proto/mace.pb.h"
#include "mace/utils/memory.h"

namespace mace {

class OpRegistry {
 public:
  OpRegistry() = default;
  virtual ~OpRegistry() = default;
  MaceStatus Register(const std::string &op_type,
                      const DeviceType device_type,
                      const DataType dt,
                      OpRegistrationInfo::OpCreator creator);

  MaceStatus Register(const OpConditionBuilder &builder);

  const std::set<DeviceType> AvailableDevices(
      const std::string &op_type, OpConditionContext *context) const;

  void GetInOutMemoryTypes(
      const std::string &op_type, OpConditionContext *context) const;

  const std::vector<DataFormat> InputsDataFormat(
      const std::string &op_type, OpConditionContext *context) const;

  std::unique_ptr<Operation> CreateOperation(
      OpConstructContext *context,
      DeviceType device_type) const;

  template<class DerivedType>
  static std::unique_ptr<Operation> DefaultCreator(
      OpConstructContext *context) {
    return make_unique<DerivedType>(context);
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<OpRegistrationInfo>>
      registry_;
  MACE_DISABLE_COPY_AND_ASSIGN(OpRegistry);
};

#define MACE_REGISTER_OP(op_registry, op_type, class_name, device, dt) \
  op_registry->Register(op_type,                                       \
                        device,                                        \
                        DataTypeToEnum<dt>::value,                     \
                        OpRegistry::DefaultCreator<class_name<device, dt>>)

#define MACE_REGISTER_OP_BY_CLASS(\
    op_registry, op_type, class_name, device, dt)  \
  op_registry->Register(op_type,                   \
                        device,                    \
                        DataTypeToEnum<dt>::value, \
                        OpRegistry::DefaultCreator<class_name>)

#ifndef MACE_REGISTER_BF16_OP
#ifdef MACE_ENABLE_BFLOAT16
#define MACE_REGISTER_BF16_OP(op_registry, op_type, class_name, device) \
    MACE_REGISTER_OP(op_registry, op_type, class_name, device, BFloat16)
#else
#define MACE_REGISTER_BF16_OP(op_registry, op_type, class_name, device)
#endif  // MACE_ENABLE_BFLOAT16
#endif  // MACE_REGISTER_BF16_OP

#ifndef MACE_REGISTER_BF16_OP_BY_CLASS
#ifdef MACE_ENABLE_BFLOAT16
#define MACE_REGISTER_BF16_OP_BY_CLASS(op_registry, op_type, \
                                       class_name, device)   \
    MACE_REGISTER_OP_BY_CLASS(op_registry, op_type,          \
                              class_name, device, BFloat16)
#else
#define MACE_REGISTER_BF16_OP_BY_CLASS(op_registry, op_type, class_name, device)
#endif  // MACE_ENABLE_BFLOAT16
#endif  // MACE_REGISTER_BF16_OP_BY_CLASS

#ifdef MACE_ENABLE_OPENCL
#define MACE_REGISTER_GPU_OP(op_registry, op_type, class_name) \
  op_registry->Register(                                       \
      op_type,                                                 \
      DeviceType::GPU,                                         \
      DT_FLOAT,                                                \
      OpRegistry::DefaultCreator<class_name<DeviceType::GPU, float>>)
#else
#define MACE_REGISTER_GPU_OP(op_registry, op_type, class_name)
#endif

#define MACE_REGISTER_OP_CONDITION(op_registry, builder) \
  op_registry->Register(builder)

}  // namespace mace

#endif  // MACE_CORE_REGISTRY_OPS_REGISTRY_H_
