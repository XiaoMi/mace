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

#include "mace/core/registry/op_registration_info.h"

#include <set>
#include <string>
#include <utility>
#include <vector>

#include "mace/core/ops/op_condition_context.h"

namespace mace {
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
  creators[key] = std::move(creator);
}

}  // namespace mace
