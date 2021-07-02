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

#include "mace/utils/mace_engine_config.h"

#include "mace/core/runtime/runtime.h"

#ifdef MACE_ENABLE_HEXAGON
#include "mace/runtimes/hexagon/dsp/hexagon_dsp_wrapper.h"
#endif  // MACE_ENABLE_HEXAGON

namespace mace {

MaceEngineCfgImpl::MaceEngineCfgImpl()
    : num_threads_(-1),
      cpu_affinity_policy_(CPUAffinityPolicy::AFFINITY_NONE),
      opencl_context_(nullptr),
      gpu_priority_hint_(GPUPriorityHint::PRIORITY_LOW),
      gpu_perf_hint_(GPUPerfHint::PERF_NORMAL),
      hexagon_corner_(HexagonNNCornerType::HEXAGON_NN_CORNER_TURBO),
      hexagon_dcvs_enable_(true),
      hexagon_latency_(100),
      accelerator_cache_policy_(AcceleratorCachePolicy::ACCELERATOR_CACHE_NONE),
      accelerator_binary_file_(""),
      accelerator_storage_file_(""),
      apu_boost_hint_(100),
      apu_preference_hint_(
        APUPreferenceHint::NEURON_PREFER_FAST_SINGLE_ANSWER) {}

void MaceEngineCfgImpl::SetRuntimeType(const RuntimeType runtime_type,
                                       const char *sub_graph_name) {
  runtime_map_[sub_graph_name] = runtime_type;
}

int MaceEngineCfgImpl::num_threads() const {
  return num_threads_;
}

CPUAffinityPolicy MaceEngineCfgImpl::cpu_affinity_policy() const {
  return cpu_affinity_policy_;
}

std::shared_ptr<OpenclContext> MaceEngineCfgImpl::opencl_context() const {
  return opencl_context_;
}

GPUPriorityHint MaceEngineCfgImpl::gpu_priority_hint() const {
  return gpu_priority_hint_;
}

GPUPerfHint MaceEngineCfgImpl::gpu_perf_hint() const {
  return gpu_perf_hint_;
}

HexagonNNCornerType MaceEngineCfgImpl::hexagon_corner() const {
  return hexagon_corner_;
}

bool MaceEngineCfgImpl::hexagon_dcvs_enable() const {
  return hexagon_dcvs_enable_;
}

int MaceEngineCfgImpl::hexagon_latency() const {
  return hexagon_latency_;
}

AcceleratorCachePolicy MaceEngineCfgImpl::accelerator_cache_policy() const {
  return accelerator_cache_policy_;
}

std::string MaceEngineCfgImpl::accelerator_binary_file() const {
  return accelerator_binary_file_;
}

std::string MaceEngineCfgImpl::accelerator_storage_file() const {
  return accelerator_storage_file_;
}

HexagonPerformanceType MaceEngineCfgImpl::hexagon_performance() const {
  return hexagon_perf_;
}

uint8_t MaceEngineCfgImpl::apu_boost_hint() const {
  return apu_boost_hint_;
}

APUPreferenceHint MaceEngineCfgImpl::apu_preference_hint() const {
  return apu_preference_hint_;
}

RuntimeType MaceEngineCfgImpl::runtime_type(
    const std::string &sub_graph_name) const {
  if (runtime_map_.count(sub_graph_name) == 0) {
    MACE_CHECK(runtime_map_.size() == 0,
               "You set the sub graph's runtime type, but we get"
               " an invalid graph: ", sub_graph_name);
    return RuntimeType::RT_NONE;
  }
  return static_cast<RuntimeType>(runtime_map_.at(sub_graph_name));
}

MaceStatus MaceEngineCfgImpl::SetOpenclContext(
    std::shared_ptr<OpenclContext> context) {
  opencl_context_ = context;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngineCfgImpl::SetGPUHints(
    GPUPerfHint perf_hint,
    GPUPriorityHint priority_hint) {
  gpu_perf_hint_ = perf_hint;
  gpu_priority_hint_ = priority_hint;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngineCfgImpl::SetCPUThreadPolicy(
    int num_threads,
    CPUAffinityPolicy policy) {
  num_threads_ = num_threads;
  cpu_affinity_policy_ = policy;
  return MaceStatus::MACE_SUCCESS;
}

MaceStatus MaceEngineCfgImpl::SetHexagonToUnsignedPD() {
  bool ret = false;
#ifdef MACE_ENABLE_HEXAGON
  ret = HexagonDSPWrapper::RequestUnsignedPD();
#endif
  return ret ? MaceStatus::MACE_SUCCESS : MaceStatus::MACE_RUNTIME_ERROR;
}

MaceStatus MaceEngineCfgImpl::SetHexagonPower(
    HexagonNNCornerType corner,
    bool dcvs_enable,
    int latency) {
  hexagon_corner_ = corner;
  hexagon_dcvs_enable_ = dcvs_enable;
  hexagon_latency_ = latency;
  bool ret = false;
#ifdef MACE_ENABLE_HEXAGON
  ret = HexagonDSPWrapper::SetPower(corner, dcvs_enable, latency);
#endif
  return ret ? MaceStatus::MACE_SUCCESS : MaceStatus::MACE_RUNTIME_ERROR;
}

MaceStatus MaceEngineCfgImpl::SetQnnPerformance(
    HexagonPerformanceType type) {
  hexagon_perf_ = type;
#ifdef MACE_ENABLE_QNN
  return MaceStatus::MACE_SUCCESS;
#else
  return MaceStatus::MACE_RUNTIME_ERROR;
#endif  // MACE_ENABLE_QNN
}

MaceStatus MaceEngineCfgImpl::SetAcceleratorCache(
    AcceleratorCachePolicy policy,
    const std::string &binary_file,
    const std::string &storage_file) {
  bool ret = false;
  accelerator_cache_policy_ = policy;
  accelerator_binary_file_ = binary_file;
  accelerator_storage_file_ = storage_file;
#ifdef MACE_ENABLE_MTK_APU
  ret = true;
#endif  // MACE_ENABLE_MTK_APU
  return ret ? MaceStatus::MACE_SUCCESS : MaceStatus::MACE_RUNTIME_ERROR;
}

MaceStatus MaceEngineCfgImpl::SetAPUHints(
    uint8_t boost_hint,
    APUPreferenceHint preference_hint) {
  bool ret = false;
  apu_boost_hint_ = boost_hint;
  apu_preference_hint_ = preference_hint;
#ifdef MACE_ENABLE_MTK_APU
  ret = true;
#endif  // MACE_ENABLE_MTK_APU
  return ret ? MaceStatus::MACE_SUCCESS : MaceStatus::MACE_RUNTIME_ERROR;
}

MaceEngineConfig::MaceEngineConfig() : impl_(new MaceEngineCfgImpl()) {}

MaceEngineConfig::~MaceEngineConfig() = default;

MaceEngineConfig::MaceEngineConfig(const DeviceType device_type)
    : impl_(new MaceEngineCfgImpl()) {
  SetRuntimeType(static_cast<RuntimeType>(device_type));
}

void MaceEngineConfig::SetRuntimeType(const RuntimeType runtime_type,
                                      const char *sub_graph_name) {
  impl_->SetRuntimeType(runtime_type, sub_graph_name);
}

MaceStatus MaceEngineConfig::SetGPUContext(
    std::shared_ptr<OpenclContext> context) {
  return impl_->SetOpenclContext(context);
}

MaceStatus MaceEngineConfig::SetGPUHints(
    GPUPerfHint perf_hint,
    GPUPriorityHint priority_hint) {
  return impl_->SetGPUHints(perf_hint, priority_hint);
}

MaceStatus MaceEngineConfig::SetCPUThreadPolicy(
    int num_threads_hint,
    CPUAffinityPolicy policy) {
  return impl_->SetCPUThreadPolicy(num_threads_hint, policy);
}

MaceStatus MaceEngineConfig::SetHexagonToUnsignedPD() {
  return impl_->SetHexagonToUnsignedPD();
}

MaceStatus MaceEngineConfig::SetHexagonPower(
    HexagonNNCornerType corner,
    bool dcvs_enable,
    int latency) {
  return impl_->SetHexagonPower(corner, dcvs_enable, latency);
}

MaceStatus MaceEngineConfig::SetQnnPerformance(
    HexagonPerformanceType type) {
  return impl_->SetQnnPerformance(type);
}

MaceStatus MaceEngineConfig::SetAcceleratorCache(
    AcceleratorCachePolicy policy,
    const std::string &binary_file,
    const std::string &storage_file) {
  return impl_->SetAcceleratorCache(policy, binary_file, storage_file);
}

MaceStatus MaceEngineConfig::SetAPUHints(
    uint8_t boost_hint,
    APUPreferenceHint preference_hint) {
  return impl_->SetAPUHints(boost_hint, preference_hint);
}

}  // namespace mace
