// Copyright 2020 The MACE Authors. All Rights Reserved.
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


#ifndef MACE_CORE_RUNTIME_RUNTIME_H_
#define MACE_CORE_RUNTIME_RUNTIME_H_

#include <memory>
#include <vector>

#include "mace/core/memory/allocator.h"
#include "mace/core/memory/buffer.h"
#include "mace/core/memory/memory_manager.h"
#include "mace/core/ops/ops_utils.h"
#include "mace/core/runtime/runtime_context.h"
#include "mace/core/types.h"
#include "mace/utils/mace_engine_config.h"
#include "mace/public/mace.h"
#include "mace/utils/thread_pool.h"

namespace mace {

constexpr const char *kNamespaceTensor = "tensor";
constexpr const char *kNamespaceScratch = "scratch";

class BaseEngine;
class NetDef;
class OpContext;
class Tensor;

class Runtime {
 public:
  explicit Runtime(RuntimeContext *runtime_context);
  virtual ~Runtime();

  // TODO(luxuhui): should be changed to RuntimeConfig
  virtual MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                          const MemoryType mem_type);

  virtual MaceStatus BeforeRun(MaceEngineCfgImpl *config);
  virtual MaceStatus AfterRun();

  virtual MaceStatus MapBuffer(Buffer *buffer, bool wait_for_finish);
  virtual MaceStatus UnMapBuffer(Buffer *buffer);
  virtual bool CanReuseBuffer(
      const Buffer *buffer, const std::vector<index_t> &shape,
      const BufferContentType content_type, const unsigned int content_param);

  virtual RuntimeType GetRuntimeType() = 0;
  virtual MemoryType GetBaseMemoryType();
  virtual MemoryType GetUsedMemoryType();

  utils::ThreadPool &thread_pool();

  MaceStatus AllocateBufferForTensor(Tensor *tensor, BufRentType rent_type,
                                     Buffer *slice_parent = nullptr,
                                     index_t offset = 0);
  void ReleaseBufferForTensor(Tensor *tensor, const BufRentType rent_type);

  std::unique_ptr<Buffer> ObtainBuffer(const MemInfo &info,
                                       BufRentType rent_type);
  void ReleaseBuffer(Buffer *buffer, BufRentType rent_type);
  void ReleaseAllBuffer(BufRentType rent_type, bool del_buf = false);

  virtual std::unique_ptr<Buffer> MakeSliceBuffer(
      const NetDef &net_def, const unsigned char *model_data,
      const index_t model_data_size) = 0;

  virtual std::vector<index_t> ComputeBufDimFromTensorDim(
      const std::vector<index_t> &dims, const MemoryType mem_type,
      const BufferContentType content_type, const unsigned int content_param);
  virtual DataType GetComputeDataType(const NetDef &net_def,
                                      const ConstTensor &const_tensor);

  virtual MemoryManager *GetMemoryManager(const MemoryType mem_type) = 0;

  void SetBufferToTensor(std::unique_ptr<Buffer> buffer, Tensor *tensor);

  // for inter buffers' release and re-allocate
  void ReleaseIntermediateBuffer(const BaseEngine *engine);
  void OnAllocateIntermediateBuffer(const BaseEngine *engine);
  void OnIntermediateBufferUsed(const BaseEngine *engine);
  bool IntermediateBufferCreated(const BaseEngine *engine) const;
  bool IntermediateBufferStable(const OpContext *op_context) const;

 protected:
  utils::ThreadPool *thread_pool_;

  // for inter buffers' release and re-allocate
  std::unordered_map<const void *, int> inter_mem_state_map_;
  bool has_ever_released_inter_mem_;  // For acceleration
};

}  // namespace mace

#endif  // MACE_CORE_RUNTIME_RUNTIME_H_
