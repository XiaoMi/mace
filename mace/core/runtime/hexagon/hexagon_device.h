// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DEVICE_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DEVICE_H_

#include <memory>
#include <utility>

#include "mace/core/device.h"
#include "mace/core/runtime/hexagon/hexagon_allocator.h"
#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#ifdef MACE_ENABLE_HEXAGON
#include "mace/core/runtime/hexagon/hexagon_dsp_wrapper.h"
#endif
#ifdef MACE_ENABLE_HTA
#include "mace/core/runtime/hexagon/hexagon_hta_wrapper.h"
#endif
#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/gpu_device.h"
#include "mace/core/runtime/opencl/gpu_runtime.h"
#endif

namespace mace {

class HexagonDevice : public CPUDevice {
 public:
  HexagonDevice(DeviceType device_type,
#ifdef MACE_ENABLE_OPENCL
                utils::ThreadPool *thread_pool,
                std::unique_ptr<GPUDevice> gpu_device);
#else
                utils::ThreadPool *thread_pool);
#endif  // MACE_ENABLE_OPENCL

#ifdef MACE_ENABLE_OPENCL
  GPURuntime *gpu_runtime() override;
#endif  // MACE_ENABLE_OPENCL
  Allocator *allocator() override;
  DeviceType device_type() const override;

 private:
  std::unique_ptr<HexagonAllocator> allocator_;
  DeviceType device_type_;
#ifdef MACE_ENABLE_OPENCL
  std::unique_ptr<GPUDevice> gpu_device_;
#endif  // MACE_ENABLE_OPENCL
};

std::unique_ptr<HexagonControlWrapper> CreateHexagonControlWrapper(
    Device *device);

}  // namespace mace
#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_DEVICE_H_
