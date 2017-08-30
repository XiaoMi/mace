//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_WORKSPACE_H_
#define MACE_CORE_WORKSPACE_H_


#include "mace/core/common.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class Workspace {
 public:
  typedef map<string, unique_ptr<Tensor>> TensorMap;

  Workspace() {}

  vector<string> Tensors() const;

  Tensor* CreateTensor(const string& name, Allocator* alloc, DataType type);

  bool RemoveTensor(const string& name);

  inline bool HasTensor(const string& name) const {
    return tensor_map_.count(name);
  }

  const Tensor* GetTensor(const string& name) const;

  Tensor* GetTensor(const string& name);

  void LoadModelTensor(const NetDef& net_def, DeviceType type);

 private:
  TensorMap tensor_map_;

 DISABLE_COPY_AND_ASSIGN(Workspace);
};

} // namespace mace


#endif // MACE_CORE_WORKSPACE_H_
