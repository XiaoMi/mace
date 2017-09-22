//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/net.h"

namespace mace {

NetBase::NetBase(const std::shared_ptr<const NetDef>& net_def,
                 Workspace* ws,
                 DeviceType type)
    : name_(net_def->name()) {}

SimpleNet::SimpleNet(const std::shared_ptr<const NetDef>& net_def,
                     Workspace* ws,
                     DeviceType type)
    : NetBase(net_def, ws, type) {
  VLOG(1) << "Constructing SimpleNet " << net_def->name();
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto& operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ":"
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    OperatorDef temp_def(operator_def);
    op = CreateOperator(temp_def, ws, type);
    operators_.emplace_back(std::move(op));
  }
}
bool SimpleNet::Run() {
  VLOG(1) << "Running net " << name_;
  for (auto& op : operators_) {
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
    if (!op->Run()) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }
    VLOG(1) << "Op " << op->debug_def().name()
            << " has shape: " << internal::MakeString(op->Output(0)->shape());
  }
  return true;
}

unique_ptr<NetBase> CreateNet(const NetDef& net_def,
                              Workspace* ws,
                              DeviceType type) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, ws, type);
}

unique_ptr<NetBase> CreateNet(const std::shared_ptr<const NetDef>& net_def,
                              Workspace* ws,
                              DeviceType type) {
  unique_ptr<NetBase> net(new SimpleNet(net_def, ws, type));
  return net;
}

}  //  namespace mace
