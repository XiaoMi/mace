// Copyright 2021 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_RUNTIMES_QNN_QNN_PERFORMANCE_H_
#define MACE_RUNTIMES_QNN_QNN_PERFORMANCE_H_

#include "mace/public/mace.h"
#include "mace/runtimes/qnn/common.h"
#include "third_party/qnn/include/HTP/QnnDspBackend.h"

namespace mace {
class QnnPerformance {
 public:
  QnnPerformance(QnnFunctionPointers* qnn_function_pointers);
  void SetPerformance(QnnGraphState state, HexagonPerformanceType type);

 private:
  void SetPowerConfig(const QnnDspPerfInfrastructure_PowerConfig_t **configs);
  void SetNormal(HexagonPerformanceType type);
  void SetRelaxed();
  void SetReleased();
  void SetInitDone(HexagonPerformanceType type);
  void SetInferenceDone(HexagonPerformanceType type);

 private:
  QnnDspBackend_PerfInfrastructure_t *infra_;
  uint32_t power_config_id_;
  MACE_DISABLE_COPY_AND_ASSIGN(QnnPerformance);
  QnnFunctionPointers* qnn_function_pointers_;
};
}  // namespace mace

#endif  // MACE_RUNTIMES_QNN_QNN_PERFORMANCE_H_
