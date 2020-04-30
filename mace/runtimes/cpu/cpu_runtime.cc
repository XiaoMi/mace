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

#include "mace/runtimes/cpu/cpu_runtime.h"

#include <vector>

#include "mace/core/memory/buffer.h"
#include "mace/core/proto/net_def_helper.h"
#include "mace/utils/memory.h"

namespace mace {

CpuRuntime::CpuRuntime(RuntimeContext *runtime_context)
    : Runtime(runtime_context) {}

MaceStatus CpuRuntime::Init(const MaceEngineCfgImpl *engine_config,
                            const MemoryType mem_type) {
  MACE_UNUSED(mem_type);
#ifdef MACE_ENABLE_QUANTIZE
  MACE_CHECK_NOTNULL(GetGemmlowpContext());
#endif  // MACE_ENABLE_QUANTIZE
  SetThreadsHintAndAffinityPolicy(engine_config->num_threads(),
                                  engine_config->cpu_affinity_policy());

  return MaceStatus::MACE_SUCCESS;
}

RuntimeType CpuRuntime::GetRuntimeType() {
  return RuntimeType::RT_CPU;
}

std::unique_ptr<Buffer> CpuRuntime::MakeSliceBuffer(
    const NetDef &net_def,
    const unsigned char *model_data, const index_t model_data_size) {
  MACE_ASSERT(model_data != nullptr && model_data_size > 0);
  if (NetDefHelper::HasHalfTensor(net_def)) {
    return nullptr;
  }

  if (!NetDefHelper::IsQuantizedModel(net_def) &&
      NetDefHelper::HasQuantizedTensor(net_def)) {
    return nullptr;
  }

  MemoryType mem_type = MemoryType::CPU_BUFFER;
  auto buffer = make_unique<Buffer>(
      mem_type, DataType::DT_UINT8, std::vector<index_t>({model_data_size}),
      static_cast<void *>(const_cast<unsigned char *>(model_data)));
  return buffer;
}

DataType CpuRuntime::GetComputeDataType(const NetDef &net_def,
                                        const ConstTensor &const_tensor) {
  if (const_tensor.data_type() == DataType::DT_HALF) {
    return DataType::DT_FLOAT;
  }
  return Runtime::GetComputeDataType(net_def, const_tensor);
}

MaceStatus CpuRuntime::SetThreadsHintAndAffinityPolicy(
    int num_threads_hint, CPUAffinityPolicy policy) {
  // get cpu frequency info
  std::vector<float> cpu_max_freqs;
  MACE_RETURN_IF_ERROR(GetCPUMaxFreq(&cpu_max_freqs));
  if (cpu_max_freqs.empty()) {
    return MaceStatus::MACE_RUNTIME_ERROR;
  }
  std::vector<size_t> cores_to_use;
  MACE_RETURN_IF_ERROR(
      mace::utils::GetCPUCoresToUse(
          cpu_max_freqs, policy, &num_threads_hint, &cores_to_use));

#ifdef MACE_ENABLE_QUANTIZE
  if (gemm_context_ != nullptr) {
    gemm_context_->set_max_num_threads(num_threads_hint);
  }
#endif  // MACE_ENABLE_QUANTIZE

  MaceStatus status = MaceStatus::MACE_SUCCESS;
  if (policy != CPUAffinityPolicy::AFFINITY_NONE) {
    status = SchedSetAffinity(cores_to_use);
    VLOG(1) << "Set affinity : " << MakeString(cores_to_use);
  }

  return status;
}

#ifdef MACE_ENABLE_QUANTIZE
gemmlowp::GemmContext *CpuRuntime::GetGemmlowpContext() {
  if (gemm_context_ == nullptr) {
    gemm_context_.reset(new gemmlowp::GemmContext(thread_pool_));
  }
  return gemm_context_.get();
}
#endif  // MACE_ENABLE_QUANTIZE

}  // namespace mace
