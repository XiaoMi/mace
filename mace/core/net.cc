//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/net.h"
#include "mace/utils/utils.h"
#ifdef __USE_OPENCL
#include "mace/core/runtime/opencl/opencl_runtime.h"
#endif

namespace mace {

NetBase::NetBase(const std::shared_ptr<const NetDef> &net_def,
                 Workspace *ws,
                 DeviceType type)
    : name_(net_def->name()) {}

SimpleNet::SimpleNet(const std::shared_ptr<const NetDef> &net_def,
                     Workspace *ws,
                     DeviceType type)
    : NetBase(net_def, ws, type), device_type_(type){
  VLOG(1) << "Constructing SimpleNet " << net_def->name();
  for (int idx = 0; idx < net_def->op_size(); ++idx) {
    const auto &operator_def = net_def->op(idx);
    VLOG(1) << "Creating operator " << operator_def.name() << ":"
            << operator_def.type();
    std::unique_ptr<OperatorBase> op{nullptr};
    OperatorDef temp_def(operator_def);
    op = CreateOperator(temp_def, ws, type);
    if (op) {
      operators_.emplace_back(std::move(op));
    }
  }
}
bool SimpleNet::Run(RunMetadata *run_metadata) {
  VLOG(1) << "Running net " << name_;
  for (auto &op : operators_) {
    VLOG(1) << "Running operator " << op->debug_def().name() << "("
            << op->debug_def().type() << ").";
    OperatorStats *op_stats = nullptr;
    if (run_metadata ) {
      if (device_type_ != DeviceType::OPENCL) {
        op_stats = run_metadata->add_op_stats();
        op_stats->set_operator_name(op->debug_def().name());
        op_stats->set_type(op->debug_def().type());
        op_stats->set_all_start_micros(NowInMicroSec());
        op_stats->set_op_start_rel_micros(NowInMicroSec() -
            op_stats->all_start_micros());
      }
    }
    if (!op->Run()) {
      LOG(ERROR) << "Operator failed: " << ProtoDebugString(op->debug_def());
      return false;
    }

    if (run_metadata) {
      if (device_type_ == DeviceType::OPENCL) {
#ifndef __USE_OPENCL
        LOG(FATAL) << "OpenCL is not supported";
#else
        OpenCLRuntime::Get()->command_queue().finish();
        op_stats = run_metadata->add_op_stats();
        op_stats->set_operator_name(op->debug_def().name());
        op_stats->set_type(op->debug_def().type());

        op_stats->set_all_start_micros(
            OpenCLRuntime::Get()->GetEventProfilingStartInfo() / 1000);
        op_stats->set_op_start_rel_micros(
            OpenCLRuntime::Get()->GetEventProfilingStartInfo() / 1000 -
            op_stats->all_start_micros());

        op_stats->set_op_end_rel_micros(
            OpenCLRuntime::Get()->GetEventProfilingEndInfo() / 1000 -
            op_stats->all_start_micros());
        op_stats->set_all_end_rel_micros(
            OpenCLRuntime::Get()->GetEventProfilingEndInfo() / 1000 -
            op_stats->all_start_micros());
#endif
      } else {
        op_stats->set_op_end_rel_micros(NowInMicroSec() -
                                        op_stats->all_start_micros());
        op_stats->set_all_end_rel_micros(NowInMicroSec() -
                                         op_stats->all_start_micros());
      }
    }
    VLOG(1) << "Op " << op->debug_def().name()
            << " has shape: " << internal::MakeString(op->Output(0)->shape());
  }
  return true;
}

unique_ptr<NetBase> CreateNet(const NetDef &net_def,
                              Workspace *ws,
                              DeviceType type) {
  std::shared_ptr<NetDef> tmp_net_def(new NetDef(net_def));
  return CreateNet(tmp_net_def, ws, type);
}

unique_ptr<NetBase> CreateNet(const std::shared_ptr<const NetDef> &net_def,
                              Workspace *ws,
                              DeviceType type) {
  unique_ptr<NetBase> net(new SimpleNet(net_def, ws, type));
  return net;
}

}  //  namespace mace
