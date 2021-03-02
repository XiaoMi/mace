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


#ifndef MACE_RUNTIMES_CPU_CPU_RUNTIME_H_
#define MACE_RUNTIMES_CPU_CPU_RUNTIME_H_

#include <memory>

#include "mace/core/runtime/runtime.h"

#ifdef MACE_ENABLE_QUANTIZE
#include "public/gemmlowp.h"
#endif  // MACE_ENABLE_QUANTIZE

namespace mace {
class CpuRuntime : public Runtime {
 public:
  explicit CpuRuntime(RuntimeContext *runtime_context);
  ~CpuRuntime() = default;

  template<typename ContextType>
  static CpuRuntime *Get(ContextType *context) {
    return static_cast<CpuRuntime *>(context->runtime());
  }

  MaceStatus Init(const MaceEngineCfgImpl *engine_config,
                  const MemoryType mem_type) override;

  RuntimeType GetRuntimeType() override;
  std::unique_ptr<Buffer> MakeSliceBuffer(
      const NetDef &net_def, const unsigned char *model_data,
      const index_t model_data_size) override;
  DataType GetComputeDataType(const NetDef &net_def,
                              const ConstTensor &const_tensor) override;

#ifdef MACE_ENABLE_QUANTIZE
  gemmlowp::GemmContext *GetGemmlowpContext();
#endif  // MACE_ENABLE_QUANTIZE

 private:
  MaceStatus SetThreadsHintAndAffinityPolicy(int num_threads_hint,
                                             CPUAffinityPolicy policy);

 private:
#ifdef MACE_ENABLE_QUANTIZE
  std::unique_ptr<gemmlowp::GemmContext> gemm_context_;
#endif  // MACE_ENABLE_QUANTIZE
};

}  // namespace mace

#endif  // MACE_RUNTIMES_CPU_CPU_RUNTIME_H_
