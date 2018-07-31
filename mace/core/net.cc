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

#include "mace/core/macros.h"
#include "mace/core/net.h"
#include "mace/utils/memory_logging.h"
#include "mace/utils/timer.h"
#include "mace/utils/utils.h"

namespace mace {

NetBase::NetBase(const std::shared_ptr<const OperatorRegistryBase> op_registry,
                 const std::shared_ptr<const NetDef> net_def,
                 Workspace *ws,
                 DeviceType type)
    : name_(net_def->name()), op_registry_(op_registry) {
  MACE_UNUSED(ws);
  MACE_UNUSED(type);
}

SerialNet::SerialNet(
    const std::shared_ptr<const OperatorRegistryBase> op_registry,
    const std::shared_ptr<const NetDef> net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode)
    : NetBase(op_registry, net_def, ws, type), device_type_(type) {
  MACE_LATENCY_LOGGER(1, "Constructing SerialNet ", net_def->name());
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto &operator_def = net_def->op(idx);
    // TODO(liuqi): refactor to add device_type to OperatorDef
    const int op_device =
        ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
            operator_def, "device", static_cast<int>(device_type_));
    if (op_device == type) {
      VLOG(3) << "Creating operator " << operator_def.name() << "("
              << operator_def.type() << ")";
      OperatorDef temp_def(operator_def);
      std::unique_ptr<OperatorBase> op(
          op_registry->CreateOperator(temp_def, ws, type, mode));
      if (op) {
        operators_.emplace_back(std::move(op));
      }
    }
  }
}

MaceStatus SerialNet::Run(RunMetadata *run_metadata) {
  MACE_MEMORY_LOGGING_GUARD();
  MACE_LATENCY_LOGGER(1, "Running net");
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    MACE_LATENCY_LOGGER(2, "Running operator ", op->debug_def().name(), "(",
                        op->debug_def().type(), "), mem_id: ",
                        MakeListString(op->debug_def().mem_id().data(),
                                       op->debug_def().mem_id().size()));
    bool future_wait = (device_type_ == DeviceType::GPU &&
                        (run_metadata != nullptr ||
                         std::distance(iter, operators_.end()) == 1));

    CallStats call_stats;
    if (future_wait) {
      StatsFuture future;
      MACE_RETURN_IF_ERROR(op->Run(&future));
      if (run_metadata != nullptr) {
        future.wait_fn(&call_stats);
      } else {
        future.wait_fn(nullptr);
      }
    } else if (run_metadata != nullptr) {
      call_stats.start_micros = NowMicros();
      MACE_RETURN_IF_ERROR(op->Run(nullptr));
      call_stats.end_micros = NowMicros();
    } else {
      MACE_RETURN_IF_ERROR(op->Run(nullptr));
    }

    if (run_metadata != nullptr) {
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
      for (auto output_shape : op->debug_def().output_shape()) {
        output_shapes.push_back({output_shape.dims().begin(),
                                 output_shape.dims().end()});
      }
      OperatorStats op_stats = {op->debug_def().name(), op->debug_def().type(),
                                output_shapes,
                                {strides, padding_type, paddings, dilations,
                                 kernels}, call_stats};
      run_metadata->op_stats.emplace_back(op_stats);
    }

    VLOG(3) << "Operator " << op->debug_def().name()
            << " has shape: " << MakeString(op->Output(0)->shape());
  }

  return MACE_SUCCESS;
}

std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistryBase> op_registry,
    const NetDef &net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(op_registry, tmp_net_def, ws, type, mode);
}

std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistryBase> op_registry,
    const std::shared_ptr<const NetDef> net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode) {
  std::unique_ptr<NetBase> net(
      new SerialNet(op_registry, net_def, ws, type, mode));
  return net;
}

}  // namespace mace
