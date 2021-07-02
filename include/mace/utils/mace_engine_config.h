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

#ifndef MACE_UTILS_MACE_ENGINE_CONFIG_H_
#define MACE_UTILS_MACE_ENGINE_CONFIG_H_

#include <memory>
#include <unordered_map>
#include <string>

#include "mace/public/mace.h"

namespace mace {

class MaceEngineCfgImpl {
 public:
  MaceEngineCfgImpl();
  ~MaceEngineCfgImpl() = default;

  void SetRuntimeType(const RuntimeType runtime_type,
                      const char *sub_graph_name);

  MaceStatus SetOpenclContext(std::shared_ptr<OpenclContext> context);

  MaceStatus SetGPUHints(GPUPerfHint perf_hint, GPUPriorityHint priority_hint);

  MaceStatus SetCPUThreadPolicy(int num_threads_hint,
                                CPUAffinityPolicy policy);

  MaceStatus SetHexagonToUnsignedPD();

  MaceStatus SetHexagonPower(HexagonNNCornerType corner,
                             bool dcvs_enable,
                             int latency);

  MaceStatus SetQnnPerformance(HexagonPerformanceType type);

  MaceStatus SetAcceleratorCache(AcceleratorCachePolicy policy,
                                 const std::string &binary_file,
                                 const std::string &storage_file);

  MaceStatus SetAPUHints(uint8_t boost_hint,
                         APUPreferenceHint preference_hint);

  int num_threads() const;

  CPUAffinityPolicy cpu_affinity_policy() const;

  std::shared_ptr<OpenclContext> opencl_context() const;

  GPUPriorityHint gpu_priority_hint() const;

  GPUPerfHint gpu_perf_hint() const;

  HexagonNNCornerType hexagon_corner() const;

  bool hexagon_dcvs_enable() const;

  int hexagon_latency() const;

  HexagonPerformanceType hexagon_performance() const;

  AcceleratorCachePolicy accelerator_cache_policy() const;

  uint8_t apu_boost_hint() const;

  APUPreferenceHint apu_preference_hint() const;

  std::string accelerator_binary_file() const;

  std::string accelerator_storage_file() const;

  RuntimeType runtime_type(const std::string &sub_graph_name) const;

 private:
  int num_threads_;
  CPUAffinityPolicy cpu_affinity_policy_;
  std::shared_ptr<OpenclContext> opencl_context_;
  GPUPriorityHint gpu_priority_hint_;
  GPUPerfHint gpu_perf_hint_;
  HexagonNNCornerType hexagon_corner_;
  bool hexagon_dcvs_enable_;
  int hexagon_latency_;
  HexagonPerformanceType hexagon_perf_;
  AcceleratorCachePolicy accelerator_cache_policy_;
  std::string accelerator_binary_file_;
  std::string accelerator_storage_file_;
  uint8_t apu_boost_hint_;
  APUPreferenceHint apu_preference_hint_;
  std::unordered_map<std::string, int> runtime_map_;
};

}  // namespace mace

#endif  // MACE_UTILS_MACE_ENGINE_CONFIG_H_
