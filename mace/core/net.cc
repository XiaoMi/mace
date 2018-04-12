//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <utility>

#include "mace/core/net.h"
#include "mace/utils/memory_logging.h"
#include "mace/utils/timer.h"
#include "mace/utils/utils.h"

namespace mace {

NetBase::NetBase(const std::shared_ptr<const OperatorRegistry> op_registry,
                 const std::shared_ptr<const NetDef> net_def,
                 Workspace *ws,
                 DeviceType type)
    : op_registry_(op_registry), name_(net_def->name()) {}

SerialNet::SerialNet(const std::shared_ptr<const OperatorRegistry> op_registry,
                     const std::shared_ptr<const NetDef> net_def,
                     Workspace *ws,
                     DeviceType type,
                     const NetMode mode)
    : NetBase(op_registry, net_def, ws, type), device_type_(type) {
  MACE_LATENCY_LOGGER(1, "Constructing SerialNet ", net_def->name());
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto &operator_def = net_def->op(idx);
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

bool SerialNet::Run(RunMetadata *run_metadata) {
  MACE_MEMORY_LOGGING_GUARD();
  MACE_LATENCY_LOGGER(1, "Running net");
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    auto &op = *iter;
    MACE_LATENCY_LOGGER(2, "Running operator ", op->debug_def().name(), "(",
                        op->debug_def().type(), ")");
    bool future_wait = (device_type_ == DeviceType::OPENCL &&
                        (run_metadata != nullptr ||
                         std::distance(iter, operators_.end()) == 1));

    bool ret;
    CallStats call_stats;
    if (future_wait) {
      StatsFuture future;
      ret = op->Run(&future);
      if (run_metadata != nullptr) {
        future.wait_fn(&call_stats);
      } else {
        future.wait_fn(nullptr);
      }
    } else if (run_metadata != nullptr) {
      call_stats.start_micros = NowMicros();
      ret = op->Run(nullptr);
      call_stats.end_micros = NowMicros();
    } else {
      ret = op->Run(nullptr);
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
        strides = op->GetRepeatedArgument<int>("strides");
        padding_type = op->GetSingleArgument<int>("padding", -1);
        paddings = op->GetRepeatedArgument<int>("padding_values");
        dilations = op->GetRepeatedArgument<int>("dilations");
        if (type.compare("Pooling") == 0) {
          kernels = op->GetRepeatedArgument<index_t>("kernels");
        } else {
          kernels = op->Input(1)->shape();
        }
      }

      OperatorStats op_stats = {op->debug_def().name(), op->debug_def().type(),
                                op->debug_def().output_shape(),
                                {strides, padding_type, paddings, dilations,
                                 kernels}, call_stats};
      run_metadata->op_stats.emplace_back(op_stats);
    }

    if (!ret) {
      LOG(ERROR) << "Operator failed: " << op->debug_def().name();
      return false;
    }

    VLOG(3) << "Operator " << op->debug_def().name()
            << " has shape: " << MakeString(op->Output(0)->shape());
  }

  return true;
}

std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistry> op_registry,
    const NetDef &net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(op_registry, tmp_net_def, ws, type, mode);
}

std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistry> op_registry,
    const std::shared_ptr<const NetDef> net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode) {
  std::unique_ptr<NetBase> net(
      new SerialNet(op_registry, net_def, ws, type, mode));
  return net;
}

}  // namespace mace
