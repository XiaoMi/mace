//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_CORE_WORKSPACE_H_
#define MACE_CORE_WORKSPACE_H_

#include "mace/core/preallocated_pooled_allocator.h"
#include "mace/core/tensor.h"
#include "mace/public/mace.h"

namespace mace {

class Workspace {
 public:
  typedef std::map<std::string, std::unique_ptr<Tensor>> TensorMap;

  Workspace() {}
  ~Workspace() {}

  Tensor *CreateTensor(const std::string &name,
                       Allocator *alloc,
                       DataType type);

  inline bool HasTensor(const std::string &name) const {
    return tensor_map_.find(name) != tensor_map_.end();
  }

  const Tensor *GetTensor(const std::string &name) const;

  Tensor *GetTensor(const std::string &name);

  std::vector<std::string> Tensors() const;

  void LoadModelTensor(const NetDef &net_def, DeviceType type);

 private:
  void CreateImageOutputTensor(const NetDef &net_def);

  TensorMap tensor_map_;

  std::unique_ptr<BufferBase> tensor_buffer_;

  PreallocatedPooledAllocator preallocated_allocator_;

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

}  // namespace mace

#endif  // MACE_CORE_WORKSPACE_H_
