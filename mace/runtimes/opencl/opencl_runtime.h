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

#ifndef MACE_RUNTIMES_OPENCL_OPENCL_RUNTIME_H_
#define MACE_RUNTIMES_OPENCL_OPENCL_RUNTIME_H_

#include <memory>
#include <vector>

#include "mace/core/memory/allocator.h"
#include "mace/core/memory/general_memory_manager.h"
#include "mace/core/runtime/runtime.h"
#include "mace/runtimes/opencl/core/opencl_executor.h"
#include "mace/runtimes/opencl/core/opencl_util.h"

namespace mace {

class OpContext;

class OpenclRuntime : public Runtime {
 public:
  explicit OpenclRuntime(RuntimeContext *runtime_context);
  virtual ~OpenclRuntime() = default;

  template<typename ContextType>
  static OpenclRuntime *Get(ContextType *context) {
    auto *runtime = context->runtime();
    MACE_CHECK(runtime->GetRuntimeType() == RuntimeType::RT_OPENCL);
    return static_cast<OpenclRuntime *>(runtime);
  }

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;
  MaceStatus BeforeRun(MaceEngineCfgImpl *config) override;
  bool CanReuseBuffer(
      const Buffer *buffer, const std::vector<index_t> &shape,
      const BufferContentType content_type,
      const unsigned int content_param) override;

  MemoryType GetUsedMemoryType() override;
  MemoryType GetBaseMemoryType() override;
  RuntimeType GetRuntimeType() override;

  OpenclExecutor *GetOpenclExecutor();
  // Only for Test
  void SetUsedMemoryType(MemoryType mem_type);

  std::unique_ptr<Buffer> MakeSliceBuffer(
      const NetDef &net_def, const unsigned char *model_data,
      const index_t model_data_size) override;

  DataType GetComputeDataType(const NetDef &net_def,
                              const ConstTensor &const_tensor) override;
  std::vector<index_t> ComputeBufDimFromTensorDim(
      const std::vector<index_t> &dims, const MemoryType mem_type,
      const BufferContentType content_type,
      const unsigned int content_param) override;

 protected:
  virtual MaceStatus CreateOpenclExecutorAndInit(
      const MaceEngineCfgImpl *engine_config);

 protected:
  std::unique_ptr<OpenclExecutor> opencl_executor_;
  MemoryType used_memory_type_;
};
}  // namespace mace

#endif  // MACE_RUNTIMES_OPENCL_OPENCL_RUNTIME_H_
