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

#include "mace/runtimes/qnn/qnn_performance.h"

#include "third_party/qnn/include/QnnTypes.h"

namespace mace {
namespace {
// Should be used on V66 and above only
constexpr int kLowerLatency = 40;
// This will limit sleep modes available while running
constexpr int kLowLatency = 100;
// This will limit sleep modes available while running
constexpr int kMediumLatency = 1000;
constexpr int kHighLatency = 2000;
}  // namespace

QnnPerformance::QnnPerformance(QnnFunctionPointers* qnn_function_pointers) {
  Qnn_ErrorHandle_t ret = QNN_SUCCESS;
  QnnBackend_PerfInfrastructure_t infra = nullptr;
  qnn_function_pointers_ = qnn_function_pointers;
  ret = qnn_function_pointers_->qnnInterface.backendGetPerfInfrastructure(&infra);
  MACE_CHECK(ret == QNN_SUCCESS,
             "QnnBackend_getPerfInfrastructure failed with error: ", ret);
  infra_ = reinterpret_cast<QnnDspBackend_PerfInfrastructure_t *>(infra);
  infra_->createPowerConfigId(&power_config_id_);
  MACE_CHECK_NOTNULL(infra_->setPowerConfig);
}

void QnnPerformance::SetPowerConfig(
    const QnnDspPerfInfrastructure_PowerConfig_t **configs) {
  Qnn_ErrorHandle_t ret = QNN_SUCCESS;
  ret = infra_->setPowerConfig(power_config_id_, configs);
  MACE_CHECK(ret == QNN_SUCCESS, "setPowerConfig failed with error: ", ret);
}

void QnnPerformance::SetNormal(HexagonPerformanceType type) {
  QnnDspPerfInfrastructure_PowerConfig_t dcvs_enable;
  dcvs_enable.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
  dcvs_enable.dcvsEnableConfig = 0;  // FALSE
  QnnDspPerfInfrastructure_PowerConfig_t sleep_latency;
  QnnDspPerfInfrastructure_PowerConfig_t dcvs_powermode;
  dcvs_powermode.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
  QnnDspPerfInfrastructure_PowerConfig_t busvcorner_min;
  QnnDspPerfInfrastructure_PowerConfig_t busvcorner_target;
  QnnDspPerfInfrastructure_PowerConfig_t busvcorner_max;
  QnnDspPerfInfrastructure_PowerConfig_t corevcorner_min;
  QnnDspPerfInfrastructure_PowerConfig_t corevcorner_target;
  QnnDspPerfInfrastructure_PowerConfig_t corevcorner_max;
  dcvs_powermode.dcvsPowerModeConfig =
      QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
  sleep_latency.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY;
  busvcorner_min.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  busvcorner_target.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  busvcorner_max.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  corevcorner_min.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  corevcorner_target.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  corevcorner_max.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  switch (type) {
    case HEXAGON_BURST:
      sleep_latency.sleepLatencyConfig = kLowerLatency;
      busvcorner_min.busVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
      busvcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
      busvcorner_max.busVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
      corevcorner_min.coreVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
      corevcorner_max.coreVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_TURBO_PLUS;
      break;
    case HEXAGON_SUSTAINED_HIGH_PERFORMANCE:
    case HEXAGON_HIGH_PERFORMANCE:
      sleep_latency.sleepLatencyConfig = kLowLatency;
      busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_TURBO;
      busvcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_TURBO;
      busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_TURBO;
      corevcorner_min.coreVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_TURBO;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_TURBO;
      corevcorner_max.coreVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_TURBO;
      break;
    case HEXAGON_LOW_BALANCED:
      sleep_latency.sleepLatencyConfig = kMediumLatency;
      busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_NOM;
      busvcorner_target.busVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_NOM;
      busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_NOM;
      corevcorner_min.coreVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_NOM;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_NOM;
      corevcorner_max.coreVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_NOM;
      break;
    case HEXAGON_BALANCED:
      sleep_latency.sleepLatencyConfig = kMediumLatency;
      busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      busvcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      corevcorner_min.coreVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      corevcorner_max.coreVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_NOM_PLUS;
      break;
    case HEXAGON_POWER_SAVER:
      sleep_latency.sleepLatencyConfig = kMediumLatency;
      busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS;
      busvcorner_target.busVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_SVS;
      busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS;
      corevcorner_min.coreVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS;
      corevcorner_max.coreVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS;
      break;
    case HEXAGON_LOW_POWER_SAVER:
      sleep_latency.sleepLatencyConfig = kMediumLatency;
      busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS2;
      busvcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS2;
      busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS2;
      corevcorner_min.coreVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS2;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS2;
      corevcorner_max.coreVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS2;
      break;
    case HEXAGON_HIGH_POWER_SAVER:
      sleep_latency.sleepLatencyConfig = kMediumLatency;
      busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      busvcorner_target.busVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      corevcorner_min.coreVoltageCornerMinConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      corevcorner_target.coreVoltageCornerTargetConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      corevcorner_max.coreVoltageCornerMaxConfig =
          DCVS_VOLTAGE_VCORNER_SVS_PLUS;
      break;
    default:
      LOG(FATAL) << "Invalid performance type to set power configs: " << type;
  }
  const QnnDspPerfInfrastructure_PowerConfig_t *configs[] = {
      &dcvs_enable,     &sleep_latency,      &dcvs_powermode,
      &busvcorner_min,  &busvcorner_target,  &busvcorner_max,
      &corevcorner_min, &corevcorner_target, &corevcorner_max,
      nullptr};
  return SetPowerConfig(configs);
}

void QnnPerformance::SetRelaxed() {
  QnnDspPerfInfrastructure_PowerConfig_t dcvs_enable;
  dcvs_enable.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
  dcvs_enable.dcvsEnableConfig = 1;  // TRUE

  QnnDspPerfInfrastructure_PowerConfig_t sleep_latency;
  sleep_latency.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_SLEEP_LATENCY;
  sleep_latency.sleepLatencyConfig = kHighLatency;

  QnnDspPerfInfrastructure_PowerConfig_t dcvs_powermode;
  dcvs_powermode.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
  dcvs_powermode.dcvsPowerModeConfig =
      QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;

  QnnDspPerfInfrastructure_PowerConfig_t busvcorner_min;
  busvcorner_min.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  busvcorner_min.busVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS2;
  QnnDspPerfInfrastructure_PowerConfig_t busvcorner_target;
  busvcorner_target.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  busvcorner_target.busVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_SVS;
  QnnDspPerfInfrastructure_PowerConfig_t busvcorner_max;
  busvcorner_max.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_BUS_VOLTAGE_CORNER;
  busvcorner_max.busVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS;

  QnnDspPerfInfrastructure_PowerConfig_t corevcorner_min;
  corevcorner_min.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  corevcorner_min.coreVoltageCornerMinConfig = DCVS_VOLTAGE_VCORNER_SVS2;
  QnnDspPerfInfrastructure_PowerConfig_t corevcorner_target;
  corevcorner_target.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  corevcorner_target.coreVoltageCornerTargetConfig = DCVS_VOLTAGE_VCORNER_SVS;
  QnnDspPerfInfrastructure_PowerConfig_t corevcorner_max;
  corevcorner_max.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_CORE_VOLTAGE_CORNER;
  corevcorner_max.coreVoltageCornerMaxConfig = DCVS_VOLTAGE_VCORNER_SVS;

  // QnnDspPerfInfrastructure_PowerConfig_t hmx_off;
  // hmx_off.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_STATE;
  // hmx_off.hmxState = QNN_DSP_PERF_INFRASTRUCTURE_HMX_OFF;

  const QnnDspPerfInfrastructure_PowerConfig_t *configs[] = {
      &dcvs_enable,
      &sleep_latency,
      &dcvs_powermode,
      &busvcorner_min,
      &busvcorner_target,
      &busvcorner_max,
      &corevcorner_min,
      &corevcorner_target,
      &corevcorner_max,
      // &hmx_off,
      nullptr};
  return SetPowerConfig(configs);
}

void QnnPerformance::SetReleased() {
  QnnDspPerfInfrastructure_PowerConfig_t dcvs_enable;
  dcvs_enable.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_ENABLE;
  dcvs_enable.dcvsEnableConfig = 1;  // TRUE

  QnnDspPerfInfrastructure_PowerConfig_t dcvs_powermode;
  dcvs_powermode.config =
      QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_POWER_MODE;
  dcvs_powermode.dcvsPowerModeConfig =
      QNN_DSP_PERF_INFRASTRUCTURE_POWERMODE_POWER_SAVER_MODE;

  // QnnDspPerfInfrastructure_PowerConfig_t hmx_off;
  // hmx_off.config = QNN_DSP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_HMX_STATE;
  // hmx_off.hmxState = QNN_DSP_PERF_INFRASTRUCTURE_HMX_OFF;

  const QnnDspPerfInfrastructure_PowerConfig_t *configs[] = {
      &dcvs_enable, &dcvs_powermode, nullptr};
  return SetPowerConfig(configs);
}

void QnnPerformance::SetInitDone(HexagonPerformanceType type) {
  switch (type) {
    case HEXAGON_BURST:
    case HEXAGON_SUSTAINED_HIGH_PERFORMANCE:
    case HEXAGON_HIGH_PERFORMANCE:
    case HEXAGON_BALANCED:
    case HEXAGON_LOW_BALANCED:
      SetRelaxed();
      break;
    case HEXAGON_POWER_SAVER:
    case HEXAGON_LOW_POWER_SAVER:
    case HEXAGON_HIGH_POWER_SAVER:
      SetReleased();
      break;
    case HEXAGON_SYSTEM_SETTINGS:
      break;
    default:
      LOG(FATAL) << "Invalid performance type: " << type;
  }
}

void QnnPerformance::SetInferenceDone(HexagonPerformanceType type) {
  switch (type) {
    case HEXAGON_BURST:
    case HEXAGON_SUSTAINED_HIGH_PERFORMANCE:
      break;
    case HEXAGON_HIGH_PERFORMANCE:
    case HEXAGON_BALANCED:
    case HEXAGON_LOW_BALANCED:
      SetRelaxed();
      break;
    case HEXAGON_POWER_SAVER:
    case HEXAGON_LOW_POWER_SAVER:
    case HEXAGON_HIGH_POWER_SAVER:
      SetReleased();
      break;
    case HEXAGON_SYSTEM_SETTINGS:
      break;
    default:
      LOG(FATAL) << "Invalid performance type: " << type;
  }
}

void QnnPerformance::SetPerformance(QnnGraphState state,
                                        HexagonPerformanceType type) {
  if (type == HEXAGON_SYSTEM_SETTINGS) return;
  switch (state) {
    case QNN_INIT_START:
    case QNN_INFERENCE_START:
      SetNormal(type);
      break;
    case QNN_INIT_DONE:
      SetInitDone(type);
      break;
    case QNN_INFERENCE_DONE:
      SetInferenceDone(type);
      break;
    default:
      LOG(FATAL) << "Invalid graph state: " << state;
  }
}

}  // namespace mace
