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

#include "mace/runtimes/qnn/qnn_wrapper.h"

#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <iomanip>

#include "mace/core/proto/arg_helper.h"
#include "mace/port/file_system.h"
#include "mace/proto/qnn_cache.pb.h"
#include "mace/runtimes/qnn/qnn_runtime.h"
#include "mace/utils/conf_util.h"
#include "third_party/qnn/include/QnnError.h"
#include "third_party/qnn/include/QnnLog.h"

namespace mace {
namespace {
std::mutex log_mutex;
void QnnLogCallback(const char *fmt,
                    QnnLog_Level_t level,
                    uint64_t timestamp,
                    va_list args) {
  const char *level_str = "";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      level_str = " ERROR ";
      break;
    case QNN_LOG_LEVEL_WARN:
      level_str = "WARNING";
      break;
    case QNN_LOG_LEVEL_INFO:
      level_str = "  INFO ";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      level_str = " DEBUG ";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      level_str = "VERBOSE";
      break;
    case QNN_LOG_LEVEL_MAX:
      level_str = "UNKNOWN";
      break;
  }

  double ms = timestamp / 1000000.0;
  // To avoid interleaved messages
  {
    std::lock_guard<std::mutex> lock(log_mutex);
    fprintf(stdout, "%8.1lfms [%-7s] ", ms, level_str);
    vfprintf(stdout, fmt, args);
    fprintf(stdout, "\n");
  }
}

template <typename IntType>
std::string IntToString(const IntType v) {
  std::stringstream stream;
  stream << v;
  return stream.str();
}

template <typename FloatType>
std::string FloatToString(const FloatType v, const int32_t precision) {
  std::stringstream stream;
  stream << std::fixed << std::setprecision(precision) << v;
  return stream.str();
}

std::once_flag backend_prepared;
void PrepareBackend() {
  LOG(INFO) << "Prepare QNN backend.";
  auto log_level = static_cast<QnnLog_Level_t>(
      GetIntEnv("MACE_QNN_LOG_LEVEL", QNN_LOG_LEVEL_WARN));
  Qnn_ErrorHandle_t ret = QnnLog_initialize(QnnLogCallback, log_level);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnLog_initialize failed with error: ", ret);

  ret = QnnBackend_initialize(nullptr);
  MACE_CHECK(ret == QNN_SUCCESS || ret == QNN_BACKEND_ERROR_ALREADY_INITIALIZED,
             "QnnBackend_initialize failed with error: ", ret);
}
}  // namespace

QnnWrapper::QnnWrapper(Runtime *runtime)
    : runtime_(runtime) {
  std::call_once(backend_prepared, PrepareBackend);
  graph_state_ = QNN_INIT_START;
  perf_ = make_unique<QnnPerformance>();
  LOG(INFO) << "QNN version: " << GetVersion();
}

std::string QnnWrapper::GetVersion() {
  Qnn_ApiVersion_t version;
  MACE_CHECK(QnnBackend_getApiVersion(&version) == QNN_SUCCESS,
             "get version error");
  std::stringstream ss;
  ss << "Core: " << version.coreApiVersion.major << "."
     << version.coreApiVersion.minor << "." << version.coreApiVersion.patch
     << ", backend: " << version.backendApiVersion.major << "."
     << version.backendApiVersion.minor << "."
     << version.backendApiVersion.patch;
  return ss.str();
}

bool QnnWrapper::SetPerformance(const HexagonPerformanceType type) {
  LOG(INFO) << "Qnn set performance: " << type;
  perf_type_ = type;
  if (perf_ != nullptr) {
    perf_->SetPerformance(graph_state_, type);
  }
  return true;
}

bool QnnWrapper::SetPerformance(const QnnGraphState state,
                                const HexagonPerformanceType type) {
  graph_state_ = state;
  perf_->SetPerformance(graph_state_, type);
  return true;
}

bool QnnWrapper::Init(const NetDef &net_def,
                      unsigned const char *model_data,
                      const index_t model_data_size,
                      const AcceleratorCachePolicy cache_policy,
                      const std::string &cache_binary_file,
                      const std::string &cache_storage_file,
                      HexagonPerformanceType perf_type) {
  Qnn_ErrorHandle_t ret;
  int64_t t0 = NowMicros();
  perf_type_ = perf_type;
  perf_->SetPerformance(QNN_INIT_START, perf_type_);

  profile_level_ =
      static_cast<QnnProfile_Level_t>(GetIntEnv("MACE_QNN_PROFILE_LEVEL", 0));
  if (profile_level_ == QNN_PROFILE_LEVEL_BASIC ||
      profile_level_ == QNN_PROFILE_LEVEL_DETAILED) {
    ret = QnnProfile_create(profile_level_, &profile_);
    MACE_CHECK(ret == QNN_SUCCESS,
               "QnnProfile_create failed with error: ", ret);
    CollectOpInfo(net_def);
  } else {
    profile_ = nullptr;
  }
  // ret = QnnBackend_registerOpPackage("libQnnHtpOpPackageExample_htp.so",
                                        // "exampleInterfaceProvider", "HTP");
  // MACE_CHECK(ret == QNN_SUCCESS,
  //              "QnnBackend_registerOpPackage HTP failed with error: ", ret);
  num_inputs_ = net_def.input_info_size();
  num_outputs_ = net_def.output_info_size();
  quantized_type_ = static_cast<DataType>(
          ProtoArgHelper::GetOptionalArg<OperatorDef, int>(
              net_def.op()[0], "T", static_cast<int>(DT_UINT8)));
  transformer_ = new QuantizeTransformer();
  transformer_->Init(runtime_, quantized_type_);
  if (cache_policy == ACCELERATOR_CACHE_LOAD) {
    MACE_CHECK(!cache_binary_file.empty());
    InitWithOfflineCache(net_def, cache_binary_file);
  } else {
    InitOnline(net_def, model_data, model_data_size);
    if (cache_policy == ACCELERATOR_CACHE_STORE) {
      MACE_CHECK(!cache_storage_file.empty());
      CacheStore(net_def, cache_storage_file);
    }
  }
  perf_->SetPerformance(QNN_INIT_DONE, perf_type_);

  int64_t t1 = NowMicros();
  VLOG(1) << t1 - t0;
  return true;
}

bool QnnWrapper::InitOnline(const NetDef &net_def,
                            unsigned const char *model_data,
                            const index_t model_data_size) {
  LOG(INFO) << "Qnn init online, it may take a very long time.";

  Qnn_ErrorHandle_t ret;
  // ret = QnnBackend_registerOpPackage("libQnnHtpOpPackageExample_cpu.so",
                                    //  "exampleInterfaceProvider", "CPU");
  // MACE_CHECK(ret == QNN_SUCCESS,
  //            "QnnBackend_registerOpPackage CPU failed with error: ", ret);
  // MACE_CHECK(model_data != nullptr && model_data_size > 0);

  ret = QnnContext_create(nullptr, &ctx_);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnContext_create failed with error: ", ret);

  MACE_CHECK(!net_def.name().empty());
  ret = QnnGraph_create(ctx_, net_def.name().c_str(), nullptr, &graph_);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnGraph_create failed with error: ", ret);

  int64_t t0 = NowMicros();

  graph_builder_.Init(&net_def, graph_, runtime_);
  graph_builder_.AddConstTensors(model_data, model_data_size);
  graph_builder_.AddModelInputs(&input_info_, &input_tensors_,
                                quantized_type_);
  graph_builder_.AddModelOutputs(&output_info_, &output_tensors_,
                                 quantized_type_);
  graph_builder_.AddOpsOutputs();
  graph_builder_.AddOps();
  int64_t t1 = NowMicros();
  LOG(INFO) << "Calling QnnGraph_finalize";
  ret = QnnGraph_finalize(graph_, nullptr, nullptr);
  if (ret != QNN_SUCCESS) {
    std::cout << std::endl;
    QnnError_Info_t err_info;
    ret = QnnError_getErrorInfo(ret, &err_info);
    MACE_CHECK(ret == QNN_SUCCESS,
               "QnnError_getErrorInfo failed with error: ", ret);
    LOG(FATAL) << "QnnGraph_finalize failed with error code: "
               << err_info.staticInfo.errorCode;
  }

  int64_t t2 = NowMicros();

  VLOG(1) << "Setup time: " << t1 - t0 << " " << t2 - t1;
  if (profile_ != nullptr) {
    CollectOpInfo(net_def);
  }
  return true;
}

bool QnnWrapper::CacheStore(const NetDef &net_def,
                            const std::string &cache_storage_file) {
  LOG(INFO) << "Storing qnn cache...";
  uint32_t binary_size = 0;
  Qnn_ErrorHandle_t ret = QnnContext_getBinarySize(ctx_, &binary_size);
  MACE_CHECK(ret == QNN_SUCCESS && binary_size != 0,
             "QnnContext_getBinarySize failed with error: ", ret);

  std::vector<uint8_t> binary_buffer(binary_size);
  uint32_t written_binary_size = 0;
  ret = QnnContext_getBinary(ctx_, binary_buffer.data(), binary_size,
                             &written_binary_size);
  MACE_CHECK(ret == QNN_SUCCESS && written_binary_size <= binary_size,
             "QnnContext_getBinary failed with error: ", ret);

  MACE_CHECK(!net_def.name().empty());

  qnn_cache::CacheContext cache_ctx;
  cache_ctx.set_graph_name(net_def.name());
  for (int i = 0; i < num_inputs_; ++i) {
    cache_ctx.add_input_ids(input_tensors_[i].id);
  }
  for (int i = 0; i < num_outputs_; ++i) {
    cache_ctx.add_output_ids(output_tensors_[i].id);
  }
  cache_ctx.set_graph_cache(binary_buffer.data(), written_binary_size);

  std::string output_buffer;
  cache_ctx.SerializeToString(&output_buffer);
  std::ofstream out_file(cache_storage_file,
                         std::ios::trunc | std::ios::binary);
  MACE_CHECK(out_file.is_open(), "Open file failed: ", cache_storage_file);
  out_file.write(output_buffer.data(), output_buffer.length());
  out_file.flush();
  out_file.close();

  LOG(INFO) << "Successfully write to file: " << cache_storage_file;

  return true;
}

bool QnnWrapper::InitWithOfflineCache(const NetDef &net_def,
                                      const std::string &cache_binary_file) {
  LOG(INFO) << "Qnn init with offline cache: " << cache_binary_file;
  MACE_CHECK(!cache_binary_file.empty());
  int64_t t0 = NowMicros();

  std::unique_ptr<mace::port::ReadOnlyMemoryRegion> buffer =
      make_unique<mace::port::ReadOnlyBufferMemoryRegion>();
  auto fs = GetFileSystem();
  MaceStatus status =
      fs->NewReadOnlyMemoryRegionFromFile(cache_binary_file.c_str(), &buffer);
  MACE_CHECK(status == MaceStatus::MACE_SUCCESS,
             "Failed to read file: ", cache_binary_file);

  qnn_cache::CacheContext cache_ctx;
  cache_ctx.ParseFromArray(buffer->data(), buffer->length());

  Qnn_ErrorHandle_t ret = QnnContext_createFromBinary(
      cache_ctx.graph_cache().data(), cache_ctx.graph_cache().length(), &ctx_,
      nullptr);
  MACE_CHECK(ret == QNN_SUCCESS,
             "QnnContext_createFromBinary failed with error: ", ret);

  ret = QnnGraph_retrieve(ctx_, cache_ctx.graph_name().c_str(), &graph_);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnGraph_retrieve failed with error: ", ret);

  graph_builder_.Init(&net_def, graph_, runtime_);
  graph_builder_.AddModelInputsFromOfflineCache(&input_info_, &input_tensors_,
                                                cache_ctx.input_ids().data(),
                                                quantized_type_);
  graph_builder_.AddModelOutputsFromOfflineCache(
      &output_info_, &output_tensors_, cache_ctx.output_ids().data(),
      quantized_type_);
  int64_t t1 = NowMicros();
  VLOG(1) << t1 - t0;
  return true;
}

bool QnnWrapper::Destroy() {
  LOG(INFO) << "Qnn teardown graph";
  Qnn_ErrorHandle_t ret;
  if (profile_ != nullptr) {
    ret = QnnProfile_free(profile_);
    MACE_CHECK(ret == QNN_SUCCESS, "QnnProfile_free failed with error: ", ret);
  }
  ret = QnnContext_free(ctx_, nullptr);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnContext_free failed with error: ", ret);

  ret = QnnBackend_terminate();
  MACE_CHECK(ret == QNN_SUCCESS,
             "QnnBackend_terminate failed with error: ", ret);
  ret = QnnLog_terminate();
  MACE_CHECK(ret == QNN_SUCCESS, "QnnLog_terminate failed with error: ", ret);

  if (profile_ != nullptr && profile_level_ == QNN_PROFILE_LEVEL_DETAILED) {
    GetPerfInfo();
  }
  return true;
}

void QnnWrapper::GetEvent(QnnProfile_EventId_t event, bool collect_op_infos) {
  QnnProfile_EventData_t event_data;
  QnnProfile_getEventData(event, &event_data);
  LOG(INFO) << "Event id: " << event << ", event type: " << event_data.type
            << ", event value: " << event_data.value
            << ", event identifier: " << event_data.identifier
            << ", event unit: " << event_data.unit;
  std::string identifier(event_data.identifier);
  if (collect_op_infos) {
    profile_info_.op_cycles.back().push_back(event_data.value);
  } else {
    if (identifier == "Accelerator (execute) time") {
      if (event_data.unit == 1)
        profile_info_.npu_time += event_data.value;
      else
        profile_info_.npu_cycle += event_data.value;
    } else if (identifier == "QNN (execute) time") {
      profile_info_.qnn_time += event_data.value;
    }
  }
}

void QnnWrapper::GetSubEvents(QnnProfile_EventId_t event) {
  const QnnProfile_EventId_t *sub_events = nullptr;
  uint32_t num_sub_events = 0;
  Qnn_ErrorHandle_t ret =
      QnnProfile_getSubEvents(event, &sub_events, &num_sub_events);
  MACE_CHECK(ret == QNN_SUCCESS,
             "QnnProfile_getSubEvents failed with error: ", ret);
  // LOG(INFO) << "Got " << num_sub_events
  //           << " sub events from event " << event;
  for (uint32_t i = 0; i < num_sub_events; ++i) {
    GetEvent(sub_events[i], true);
  }
}

void QnnWrapper::CollectOpInfo(const NetDef &net_def) {
  profile_info_.op_infos.push_back({"input", "input"});
  profile_info_.op_infos.push_back({"input", "input"});
  profile_info_.output_shapes.push_back({});
  profile_info_.output_shapes.push_back({});
  for (const OperatorDef &op : net_def.op()) {
    if (op.type() == "Dequantize" || op.type() == "Quantize") {
      continue;
    }
    profile_info_.output_shapes.emplace_back(
        op.output_shape()[0].dims().begin(),
        op.output_shape()[0].dims().end());
    profile_info_.op_infos.push_back({op.name(), op.type()});
  }
  profile_info_.output_shapes.push_back({});
  profile_info_.op_infos.push_back({"output", "output"});
}

void QnnWrapper::CollectPerfInfo() {
  profile_info_.op_cycles.push_back({});
  const QnnProfile_EventId_t *events = nullptr;
  uint32_t num_events = 0;
  QnnProfile_getEvents(profile_, &events, &num_events);
  // LOG(INFO) << "Got " << num_events << " profile events";
  for (uint32_t i = 0; i < num_events; ++i) {
    GetEvent(events[i], false);
    GetSubEvents(events[i]);
  }
}

void QnnWrapper::GetPerfInfo() {
  const unsigned int n_rounds = profile_info_.op_cycles.size();
  const unsigned int n_items = profile_info_.op_cycles.back().size();
  profile_info_.npu_time /= n_rounds;
  profile_info_.npu_cycle /= n_rounds;
  profile_info_.qnn_time /= n_rounds;
  const float mhz_freq = profile_info_.npu_cycle / profile_info_.npu_time;
  const std::string run_order_title = "Sort by Run Order";
  const std::vector<std::string> run_order_header = {
      "Op Id", "Op Name", "Op Type", "Executions",
      "Duration(us)", "Output Shape"
  };
  std::vector<std::vector<std::string>> run_order_data;
  std::unordered_map<std::string, std::pair<int, float>> op_type_counters;
  std::vector<std::string> op_types;

  for (unsigned int i = 1; i < n_items; ++i) {
    const std::string op_name = profile_info_.op_infos[i].first;
    const std::string op_type = profile_info_.op_infos[i].second;

    std::vector<std::string> tuple;
    tuple.push_back(IntToString(i));
    tuple.push_back(op_name);
    tuple.push_back(op_type);
    tuple.push_back(IntToString(n_rounds));
    unsigned int op_cycle = 0;
    for (unsigned int j = 0; j < n_rounds; ++j) {
      op_cycle += profile_info_.op_cycles[j][i];
    }
    float op_time = op_cycle / mhz_freq;
    unsigned int avg_op_time = op_time / n_rounds;
    tuple.push_back(FloatToString(avg_op_time, 3));
    std::string output_shape = MakeString(profile_info_.output_shapes[i]);
    tuple.push_back(output_shape);
    run_order_data.emplace_back(tuple);
    if (op_type_counters.find(op_type) == op_type_counters.end()) {
      op_type_counters[op_type] = {0, 0.0};
      op_types.push_back(op_type);
    }
    ++op_type_counters[op_type].first;
    op_type_counters[op_type].second += avg_op_time;
  }
  std::sort(op_types.begin(), op_types.end(),
            [&](const std::string &lhs, const std::string &rhs) {
              return op_type_counters[lhs].second
                  > op_type_counters[rhs].second;
            });

  std::string duration_title = "Sort by Duration";
  const std::vector<std::string> duration_header = {
      "Op Type", "Times", "Duration(us)"
  };
  std::vector<std::vector<std::string>> duration_data;
  for (auto &op_type : op_types) {
    auto op_type_counter = op_type_counters[op_type];
    std::vector<std::string> tuple;
    tuple.push_back(op_type);
    tuple.push_back(std::to_string(op_type_counter.first));
    tuple.push_back(std::to_string(op_type_counter.second));
    duration_data.emplace_back(tuple);
  }
  LOG(INFO) << mace::string_util::StringFormatter::Table(
      run_order_title, run_order_header, run_order_data);
  LOG(INFO) << mace::string_util::StringFormatter::Table(
      duration_title, duration_header, duration_data);
  LOG(INFO) << "Qnn time: " << profile_info_.qnn_time / 1000.0f << "ms.";
  LOG(INFO) << "NPU time: " << profile_info_.npu_time  / 1000.0f << "ms.";
  LOG(INFO) << "NPU frequency: " << mhz_freq << "MHz.";
}

bool QnnWrapper::Run(const std::map<std::string, Tensor *> &input_tensors,
                     std::map<std::string, Tensor *> *output_tensors) {
  VLOG(1) << "Execute graph";

  MACE_CHECK(num_inputs_ == static_cast<int>(input_tensors.size()),
             "Wrong inputs num");
  MACE_CHECK(num_outputs_ == static_cast<int>(output_tensors->size()),
             "Wrong outputs num");
  for (int i = 0; i < num_inputs_; ++i) {
    auto input_tensor = input_tensors.at(input_info_[i].name);
    input_tensor->SetScale(input_info_[i].scale);
    input_tensor->SetZeroPoint(input_info_[i].zero_point);

    auto quantized_tensor = input_info_[i].quantized_tensor.get();
    MACE_CHECK_SUCCESS(transformer_->Quantize(
        input_tensor, quantized_tensor));
    Tensor::MappingGuard input_guard(quantized_tensor);

    if (quantized_type_ == DT_UINT16) {
      input_tensors_[i].clientBuf.data =
        const_cast<uint16_t *>(quantized_tensor->data<uint16_t>());
    } else {
      input_tensors_[i].clientBuf.data =
        const_cast<uint8_t *>(quantized_tensor->data<uint8_t>());
    }
    input_tensors_[i].clientBuf.dataSize = quantized_tensor->raw_size();
  }

  for (int i = 0; i < num_outputs_; ++i) {
    auto quantized_tensor = output_info_[i].quantized_tensor.get();
    Tensor::MappingGuard output_guard(quantized_tensor);

    if (quantized_type_ == DT_UINT16) {
      output_tensors_[i].clientBuf.data =
        quantized_tensor->mutable_data<uint16_t>();
    } else {
      output_tensors_[i].clientBuf.data =
        quantized_tensor->mutable_data<uint8_t>();
    }
    output_tensors_[i].clientBuf.dataSize = quantized_tensor->raw_size();
  }

  perf_->SetPerformance(QNN_INFERENCE_START, perf_type_);
  Qnn_ErrorHandle_t ret = QnnGraph_execute(
      graph_, input_tensors_.data(), input_tensors_.size(),
      output_tensors_.data(), output_tensors_.size(), profile_, nullptr);
  MACE_CHECK(ret == QNN_SUCCESS, "QnnGraph_execute failed with error: ", ret);
  perf_->SetPerformance(QNN_INFERENCE_DONE, perf_type_);

  for (int i = 0; i < num_outputs_; ++i) {
    auto output_tensor = output_tensors->at(output_info_[i].name);
    MaceStatus st = transformer_->Dequantize(
        output_info_[i].quantized_tensor.get(), output_tensor);
  }
  if (profile_ != nullptr && !profile_info_.is_warmup) {
    CollectPerfInfo();
  }
  if (profile_info_.is_warmup)
    profile_info_.is_warmup = false;
  return true;
}

}  // namespace mace
