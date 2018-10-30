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

#include <utility>
#include <algorithm>
#include <limits>

#include "mace/core/future.h"
#include "mace/core/macros.h"
#include "mace/core/net.h"
#include "mace/core/op_context.h"
#include "mace/public/mace.h"
#include "mace/utils/memory_logging.h"
#include "mace/utils/timer.h"
#include "mace/utils/utils.h"

namespace mace {

SerialNet::SerialNet(OpDefRegistryBase *op_def_registry,
                     const OpRegistryBase *op_registry,
                     const NetDef *net_def,
                     Workspace *ws,
                     Device *target_device,
                     const NetMode mode)
    : NetBase(),
      ws_(ws),
      target_device_(target_device),
      cpu_device_(
          new CPUDevice(target_device->cpu_runtime()->num_threads(),
                        target_device->cpu_runtime()->policy(),
                        target_device->cpu_runtime()->use_gemmlowp())) {
  MACE_LATENCY_LOGGER(1, "Constructing SerialNet");
  // Register Operations
  MaceStatus status;
  for (int idx = 0; idx < net_def->op_types_size(); ++idx) {
    status = op_def_registry->Register(net_def->op_types(idx));
    MACE_CHECK(status == MaceStatus::MACE_SUCCESS, status.information());
  }
  // Create Operations
  operators_.clear();
  const OpRegistrationInfo *info;
  DeviceType target_device_type = target_device_->device_type();
  OpConstructContext construct_context(ws_);
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto &operator_def = net_def->op(idx);
    // Create the Operation
    const int op_device =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            operator_def, "device", static_cast<int>(target_device_type));
    if (op_device == target_device_type) {
      // Find op registration information
      status = op_def_registry->Find(operator_def.type(), &info);
      MACE_CHECK(status == MaceStatus::MACE_SUCCESS, status.information());
      // Get available devices (sorted based on priority)
      OperatorDef temp_def(operator_def);
      auto available_devices = info->device_place_func_();
      // Find the device type to run the op.
      // If the target_device_type in available devices, use target_device_type,
      // otherwise, fallback to the first device (top priority).
      DeviceType device_type = available_devices[0];
      construct_context.set_device(cpu_device_);
      for (auto device : available_devices) {
        if (device == target_device_type) {
          device_type = target_device_type;
          construct_context.set_device(target_device_);
          break;
        }
      }
      temp_def.set_device_type(device_type);
      construct_context.set_operator_def(&temp_def);
      std::unique_ptr<Operation> op(
          op_registry->CreateOperation(&construct_context, device_type, mode));
      if (op) {
        operators_.emplace_back(std::move(op));
      }
    }
  }
}

MaceStatus SerialNet::Init() {
  // TODO(liuqi): where to do memory reuse.
  MACE_LATENCY_LOGGER(1, "Initializing SerialNet");
  OpInitContext init_context(ws_);
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    if (device_type == target_device_->device_type()) {
      init_context.set_device(target_device_);
    } else {
      init_context.set_device(cpu_device_);
    }
    // Initialize the operation
    MACE_RETURN_IF_ERROR(op->Init(&init_context));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialNet::Run(RunMetadata *run_metadata) {
  // TODO(liuqi): In/Out Buffer Transform
  MACE_MEMORY_LOGGING_GUARD();
  MACE_LATENCY_LOGGER(1, "Running net");
  OpContext context(ws_, cpu_device_);
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    MACE_LATENCY_LOGGER(2, "Running operator ", op->debug_def().name(),
                        "<", device_type, ", ", op->debug_def().type(), ">",
                        ". mem_id: ",
                        MakeListString(op->debug_def().mem_id().data(),
                                       op->debug_def().mem_id().size()));
    if (device_type == target_device_->device_type()) {
      context.set_device(target_device_);
    } else {
      context.set_device(cpu_device_);
    }

    CallStats call_stats;
    if (run_metadata == nullptr) {
      MACE_RETURN_IF_ERROR(op->Run(&context));
    } else {
      if (device_type == DeviceType::CPU) {
        call_stats.start_micros = NowMicros();
        MACE_RETURN_IF_ERROR(op->Run(&context));
        call_stats.end_micros = NowMicros();
      } else if (device_type == DeviceType::GPU) {
        StatsFuture future;
        context.set_future(&future);
        MACE_RETURN_IF_ERROR(op->Run(&context));
        future.wait_fn(&call_stats);
      }

      // Record run metadata
      std::vector<int> strides;
      int padding_type = -1;
      std::vector<int> paddings;
      std::vector<int> dilations;
      std::vector<index_t> kernels;
      std::string type = op->debug_def().type();

      if (type.compare("Conv2D") == 0 ||
          type.compare("FusedConv2D") == 0 ||
          type.compare("DepthwiseConv2d") == 0 ||
          type.compare("Pooling") == 0) {
        strides = op->GetRepeatedArgs<int>("strides");
        padding_type = op->GetOptionalArg<int>("padding", -1);
        paddings = op->GetRepeatedArgs<int>("padding_values");
        dilations = op->GetRepeatedArgs<int>("dilations");
        if (type.compare("Pooling") == 0) {
          kernels = op->GetRepeatedArgs<index_t>("kernels");
        } else {
          kernels = op->Input(1)->shape();
        }
      }

      std::vector<std::vector<int64_t>> output_shapes;
      for (auto output : op->Outputs()) {
        output_shapes.push_back(output->shape());
      }
      OperatorStats op_stats = {op->debug_def().name(), op->debug_def().type(),
                                output_shapes,
                                {strides, padding_type, paddings, dilations,
                                 kernels}, call_stats};
      run_metadata->op_stats.emplace_back(op_stats);
    }

    VLOG(3) << "Operator " << op->debug_def().name()
            << " has shape: " << MakeString(op->Output(0)->shape());

    if (EnvEnabled("MACE_LOG_TENSOR_RANGE")) {
      for (int i = 0; i < op->OutputSize(); ++i) {
        if (op->debug_def().quantize_info_size() == 0) {
          int data_type = op->GetOptionalArg("T", static_cast<int>(DT_FLOAT));
          if (data_type == static_cast<int>(DT_FLOAT)) {
            float max_v = std::numeric_limits<float>::lowest();
            float min_v = std::numeric_limits<float>::max();
            Tensor::MappingGuard guard(op->Output(i));
            const float *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
              max_v = std::max(max_v, output_data[j]);
              min_v = std::min(min_v, output_data[j]);
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                      << "@@" << min_v << "," << max_v;
          }
        } else {
          const int bin_size = 2048;
          for (int ind = 0; ind < op->debug_def().quantize_info_size(); ++ind) {
            float min_v = op->debug_def().quantize_info(ind).minval();
            float max_v = op->debug_def().quantize_info(ind).maxval();
            std::vector<int> bin_distribution(bin_size, 0);
            float bin_v = (max_v - min_v) / bin_size;
            Tensor::MappingGuard guard(op->Output(i));
            const float *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
                int ind = static_cast<int>((output_data[j] - min_v) / bin_v);
                if (ind < 0)
                  ind = 0;
                else if (ind > bin_size-1)
                  ind = bin_size-1;
                bin_distribution[ind]++;
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                        << "@@" << min_v << "," << max_v<< "@@"
                        << MakeString(bin_distribution);
          }
        }
      }
    }
  }

  return MaceStatus::MACE_SUCCESS;
}
}  // namespace mace
