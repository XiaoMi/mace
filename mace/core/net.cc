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

#include <algorithm>
#include <limits>
#include <set>
#include <unordered_set>
#include <utility>

#include "mace/core/future.h"
#include "mace/core/memory_optimizer.h"
#include "mace/core/net.h"
#include "mace/core/op_context.h"
#include "mace/public/mace.h"
#include "mace/port/env.h"
#include "mace/utils/conf_util.h"
#include "mace/utils/logging.h"
#include "mace/utils/macros.h"
#include "mace/utils/math.h"
#include "mace/utils/memory.h"
#include "mace/utils/timer.h"

namespace mace {

SerialNet::SerialNet(const OpRegistryBase *op_registry,
                     const NetDef *net_def,
                     Workspace *ws,
                     Device *target_device,
                     MemoryOptimizer *mem_optimizer)
    : NetBase(),
      ws_(ws),
      target_device_(target_device),
      cpu_device_(
          make_unique<CPUDevice>(
              target_device->cpu_runtime()->num_threads(),
              target_device->cpu_runtime()->policy(),
              &target_device->cpu_runtime()->thread_pool())) {
  MACE_LATENCY_LOGGER(1, "Constructing SerialNet");

#ifdef MACE_ENABLE_OPENCL
  // used for memory optimization
  std::unordered_map<std::string, MemoryType> output_mem_map;
#endif  // MACE_ENABLE_OPENCL

  OpConstructContext construct_context(ws_);
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    std::shared_ptr<OperatorDef> op_def(new OperatorDef(net_def->op(idx)));
    // Create operation
    auto op_device_type = static_cast<DeviceType>(op_def->device_type());
    if (op_device_type == target_device_->device_type()) {
      construct_context.set_device(target_device_);
    } else if (op_device_type == DeviceType::CPU) {
      construct_context.set_device(cpu_device_.get());
    } else {
      LOG(FATAL) << "Encounter unexpected error: "
                 << op_device_type << " vs " << target_device_->device_type();
    }
    construct_context.set_operator_def(op_def);

    auto op = op_registry->CreateOperation(&construct_context,
                                           op_device_type);
    operators_.emplace_back(std::move(op));
    // where to do graph reference count.
    mem_optimizer->UpdateTensorRef(op_def.get());

#ifdef MACE_ENABLE_OPENCL
    if (target_device_->device_type() == DeviceType::GPU) {
      // update the map : output_tensor -> MemoryType
      MemoryType out_mem_type =
          static_cast<MemoryType>(
              ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                  net_def->op(idx), OutputMemoryTypeTagName(),
                  static_cast<int>(MemoryType::CPU_BUFFER)));
      for (int out_idx = 0; out_idx < op_def->output_size(); ++out_idx) {
        output_mem_map[op_def->output(out_idx)] = out_mem_type;
      }
    }
#endif  // MACE_ENABLE_OPENCL
  }
  // Update output tensor reference
  for (auto &output_info : net_def->output_info()) {
    mem_optimizer->UpdateTensorRef(output_info.name());
  }

  // Do memory optimization
  for (auto &op : operators_) {
    VLOG(2) << "Operator " << op->debug_def().name() << "<" << op->device_type()
            << ", " << op->debug_def().type() << ">";
#ifdef MACE_ENABLE_OPENCL
    mem_optimizer->Optimize(op->operator_def().get(), &output_mem_map);
#else
    mem_optimizer->Optimize(op->operator_def().get());
#endif  // MACE_ENABLE_OPENCL
  }
  VLOG(1) << mem_optimizer->DebugInfo();
}

MaceStatus SerialNet::Init() {
  MACE_LATENCY_LOGGER(1, "Initializing SerialNet");
  OpInitContext init_context(ws_);
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    if (device_type == target_device_->device_type()) {
      init_context.set_device(target_device_);
    } else {
      init_context.set_device(cpu_device_.get());
    }
    // Initialize the operation
    MACE_RETURN_IF_ERROR(op->Init(&init_context));
  }
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus SerialNet::Run(RunMetadata *run_metadata) {
  MACE_MEMORY_LOGGING_GUARD();
  MACE_LATENCY_LOGGER(1, "Running net");
  OpContext context(ws_, cpu_device_.get());
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    DeviceType device_type = op->device_type();
    MACE_LATENCY_LOGGER(1, "Running operator ", op->debug_def().name(),
                        "<", device_type, ", ", op->debug_def().type(),
                        ", ",
                        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
                            op->debug_def(), "T", static_cast<int>(DT_FLOAT)),
                        ">");
    if (device_type == target_device_->device_type()) {
      context.set_device(target_device_);
    } else {
      context.set_device(cpu_device_.get());
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
          type.compare("Deconv2D") == 0 ||
          type.compare("DepthwiseConv2d") == 0 ||
          type.compare("DepthwiseDeconv2d") == 0 ||
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
      } else if (type.compare("MatMul") == 0) {
        bool transpose_a = op->GetOptionalArg<bool>("transpose_a", false);
        kernels = op->Input(0)->shape();
        if (transpose_a) {
          std::swap(kernels[kernels.size() - 2], kernels[kernels.size() - 1]);
        }
      } else if (type.compare("FullyConnected") == 0) {
        kernels = op->Input(1)->shape();
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

    if (EnvConfEnabled("MACE_LOG_TENSOR_RANGE")) {
      for (int i = 0; i < op->OutputSize(); ++i) {
        if (op->debug_def().quantize_info_size() == 0) {
          int data_type = op->GetOptionalArg("T", static_cast<int>(DT_FLOAT));
          if (data_type == static_cast<int>(DT_FLOAT)) {
            float max_v = std::numeric_limits<float>::lowest();
            float min_v = std::numeric_limits<float>::max();
            Tensor::MappingGuard guard(op->Output(i));
            auto *output_data = op->Output(i)->data<float>();
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
            auto *output_data = op->Output(i)->data<float>();
            for (index_t j = 0; j < op->Output(i)->size(); ++j) {
              int index = static_cast<int>((output_data[j] - min_v) / bin_v);
              if (index < 0)
                index = 0;
              else if (index > bin_size - 1)
                index = bin_size - 1;
              bin_distribution[index]++;
            }
            LOG(INFO) << "Tensor range @@" << op->debug_def().output(i)
                      << "@@" << min_v << "," << max_v << "@@"
                      << MakeString(bin_distribution);
          }
        }
      }
    }
  }

  return MaceStatus::MACE_SUCCESS;
}
}  // namespace mace
