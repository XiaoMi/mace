//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/net.h"
#include "mace/core/operator.h"
#include "mace/core/workspace.h"
#include "mace/utils/utils.h"

namespace mace {

NetBase::NetBase(const std::shared_ptr<const NetDef> &net_def,
                 Workspace *ws,
                 DeviceType type)
    : name_(net_def->name()) {}

SimpleNet::SimpleNet(const std::shared_ptr<const NetDef> &net_def,
                     Workspace *ws,
                     DeviceType type,
                     const NetMode mode)
    : NetBase(net_def, ws, type), device_type_(type){
  VLOG(1) << "Constructing SimpleNet " << net_def->name();
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto &operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ":"
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    OperatorDef temp_def(operator_def);
    op = CreateOperator(temp_def, ws, type, mode);
    if (op) {
      operators_.emplace_back(std::move(op));
    }
  }
}

bool SimpleNet::Run(RunMetadata *run_metadata) {
  VLOG(1) << "Running net " << name_;
  for (auto iter = operators_.begin(); iter != operators_.end(); ++iter) {
    bool future_wait = (device_type_ == DeviceType::OPENCL &&
                        (run_metadata != nullptr ||
                         std::distance(iter, operators_.end()) == 1));
    auto &op = *iter;
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";

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
      call_stats.start_micros = NowInMicroSec();
      ret = op->Run(nullptr);
      call_stats.end_micros = NowInMicroSec();
    } else {
      ret = op->Run(nullptr);
    }

    if (run_metadata != nullptr) {
      OperatorStats op_stats = { op->debug_def().name(),
                                 op->debug_def().type(),
                                 call_stats };
      run_metadata->op_stats.emplace_back(op_stats);
    }

    if (!ret) {
      LOG(ERROR) << "Operator failed: " << op->debug_def().name();
      return false;
    }

    VLOG(1) << "Op " << op->debug_def().name()
            << " has shape: " << internal::MakeString(op->Output(0)->shape());
  }

  return true;
}

unique_ptr<NetBase> CreateNet(const NetDef &net_def,
                              Workspace *ws,
                              DeviceType type,
                              const NetMode mode) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, ws, type, mode);
}

unique_ptr<NetBase> CreateNet(const std::shared_ptr<const NetDef> &net_def,
                              Workspace *ws,
                              DeviceType type,
                              const NetMode mode) {
  unique_ptr<NetBase> net(new SimpleNet(net_def, ws, type, mode));
  return net;
}

}  //  namespace mace
