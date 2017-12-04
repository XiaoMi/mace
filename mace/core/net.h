//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_NET_H_
#define MACE_CORE_NET_H_

#include "mace/core/common.h"
#include "mace/core/operator.h"
#include "mace/core/workspace.h"
#include "mace/proto/mace.pb.h"
#include "mace/proto/stats.pb.h"

namespace mace {

class NetBase {
 public:
  NetBase(const std::shared_ptr<const NetDef> &net_def,
          Workspace *ws,
          DeviceType type);
  virtual ~NetBase() noexcept {}

  virtual bool Run(RunMetadata *run_metadata = nullptr) = 0;

  const string &Name() const { return name_; }

 protected:
  string name_;

  DISABLE_COPY_AND_ASSIGN(NetBase);
};

class SimpleNet : public NetBase {
 public:
  SimpleNet(const std::shared_ptr<const NetDef> &net_def,
            Workspace *ws,
            DeviceType type,
            const OpMode mode = OpMode::NORMAL);

  bool Run(RunMetadata *run_metadata = nullptr) override;

 protected:
  vector<unique_ptr<OperatorBase> > operators_;
  DeviceType device_type_;

  DISABLE_COPY_AND_ASSIGN(SimpleNet);
};

unique_ptr<NetBase> CreateNet(const NetDef &net_def,
                              Workspace *ws,
                              DeviceType type,
                              const OpMode mode = OpMode::NORMAL);
unique_ptr<NetBase> CreateNet(const std::shared_ptr<const NetDef> &net_def,
                              Workspace *ws,
                              DeviceType type,
                              const OpMode mode = OpMode::NORMAL);

}  //  namespace mace

#endif  // MACE_CORE_NET_H_
