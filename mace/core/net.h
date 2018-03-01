//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_NET_H_
#define MACE_CORE_NET_H_

#include "mace/core/operator.h"
#include "mace/public/mace.h"

namespace mace {

class RunMetadata;
class OperatorBase;
class Workspace;

class NetBase {
 public:
  NetBase(const std::shared_ptr<const OperatorRegistry> op_registry,
          const std::shared_ptr<const NetDef> net_def,
          Workspace *ws,
          DeviceType type);
  virtual ~NetBase() noexcept {}

  virtual bool Run(RunMetadata *run_metadata = nullptr) = 0;

  const string &Name() const { return name_; }

 protected:
  string name_;
  const std::shared_ptr<const OperatorRegistry> op_registry_;

  DISABLE_COPY_AND_ASSIGN(NetBase);
};

class SimpleNet : public NetBase {
 public:
  SimpleNet(const std::shared_ptr<const OperatorRegistry> op_registry,
            const std::shared_ptr<const NetDef> net_def,
            Workspace *ws,
            DeviceType type,
            const NetMode mode = NetMode::NORMAL);

  bool Run(RunMetadata *run_metadata = nullptr) override;

 protected:
  std::vector<std::unique_ptr<OperatorBase> > operators_;
  DeviceType device_type_;

  DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistry> op_registry,
    const NetDef &net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode = NetMode::NORMAL);
std::unique_ptr<NetBase> CreateNet(
    const std::shared_ptr<const OperatorRegistry> op_registry,
    const std::shared_ptr<const NetDef> net_def,
    Workspace *ws,
    DeviceType type,
    const NetMode mode = NetMode::NORMAL);

}  // namespace mace

#endif  // MACE_CORE_NET_H_
