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

#include "mace/core/runtime/hexagon/hexagon_hta_wrapper.h"
#include <sys/types.h>
#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "mace/core/runtime/hexagon/hexagon_hta_ops.h"
#include "mace/core/runtime/hexagon/hexagon_hta_transformer.h"
#include "mace/core/types.h"
#include "mace/utils/memory.h"
#include "third_party/hta/hta_hexagon_api.h"
namespace mace {

namespace {
int GetHTAEnv(const std::string &name, int default_value) {
  int value = default_value;
  std::string env_str;
  MaceStatus status = GetEnv(name.c_str(), &env_str);
  if (status == MaceStatus::MACE_SUCCESS && !env_str.empty()) {
    value = std::atoi(env_str.c_str());
  }
  return value;
}

// Print the API logs to standard output.
void HtaApiLog(hexagon_hta_nn_nn_id id,
               const char *api_op,
               const char *const format,
                 ...) {
  va_list arg;
  va_start(arg, format);
  if (api_op != NULL) {
    printf("Graph ID: %d\t", id);
  }
  vfprintf(stdout, format, arg);
  va_end(arg);
}
// Print the performance stats to standard output.
void HtaPerformanceLog(int log_level,
                       uint32_t network_handle,
                       uint32_t thread_id,
                       const char *const format,
                       ...) {
  va_list arg;
  va_start(arg, format);
  printf("Log Level: %d, Network Handle: %d, Thread ID: %d - ", log_level,
         network_handle, thread_id);
  vfprintf(stdout, format, arg);
  va_end(arg);
}
}  // namespace

HexagonHTAWrapper::HexagonHTAWrapper(Device *device)
    : allocator_(device->allocator()),
#ifdef MACE_ENABLE_OPENCL
      transformer_(make_unique<HexagonHTATranformer<GPU>>()) {
#else
      transformer_(make_unique<HexagonHTATranformer<CPU>>()) {
#endif
  transformer_->Init(device);
}

int HexagonHTAWrapper::GetVersion() {
  int version;
  MACE_CHECK(hexagon_hta_nn_version(&version) == 0, "get version error");
  return version;
}

bool HexagonHTAWrapper::Config() {
  LOG(INFO) << "HTA config";
  MACE_CHECK(hexagon_hta_nn_config() == 0, "hexagon config error");
  return true;
}

bool HexagonHTAWrapper::Init() {
  LOG(INFO) << "Hexagon init";
  MACE_CHECK(hexagon_hta_nn_init(&nn_id_) == 0, "hexagon_nn_init failed");
  ResetPerfInfo();

  int ret;
  int power_level = GetHTAEnv("MACE_HTA_POWER_LEVEL", -1);
  if (power_level != -1) {
    ret = hexagon_hta_nn_set_config_params(nn_id_, HTA_NN_CONFIG_POWER_LEVEL,
                                           &power_level, sizeof(power_level));
    LOG(INFO) << "HTA_NN_CONFIG_POWER_LEVEL: " << power_level
              << " returns: " << ret;
  }

  int is_compress = GetHTAEnv("MACE_HTA_BANDWIDTH_COMPRESSION", 1);
  if (is_compress) {
    ret = hexagon_hta_nn_set_config_params(nn_id_,
                                           HTA_NN_CONFIG_BANDWIDTH_COMPRESSION,
                                           &is_compress, sizeof(is_compress));
    LOG(INFO) << "HTA_NN_CONFIG_BANDWIDTH_COMPRESSION: " << is_compress
              << " returns: " << ret;
  }

  if (VLOG_IS_ON(2)) {
    ret = hexagon_hta_nn_set_config_params(
        nn_id_, HTA_NN_CONFIG_PERFORMANCE_LOG,
        reinterpret_cast<void *>(&HtaPerformanceLog),
        sizeof(&HtaPerformanceLog));
    MACE_CHECK(ret == 0, "HTA_NN_CONFIG_PERFORMANCE_LOG returns: " , ret);
  }

  if (VLOG_IS_ON(3)) {
    ret = hexagon_hta_nn_set_config_params(nn_id_, HTA_NN_CONFIG_API_LOG,
                                           reinterpret_cast<void *>(&HtaApiLog),
                                           sizeof(&HtaApiLog));
    MACE_CHECK(ret == 0, "HTA_NN_CONFIG_API_LOG returns: ", ret);
  }

  return true;
}

bool HexagonHTAWrapper::Finalize() {
  LOG(INFO) << "Hexagon finalize";
  return true;
}

bool HexagonHTAWrapper::SetupGraph(const NetDef &net_def,
                                   unsigned const char *model_data) {
  LOG(INFO) << "Hexagon setup graph";

  int64_t t0 = NowMicros();

  // const node
  for (const ConstTensor &const_tensor : net_def.tensors()) {
    std::vector<int> tensor_shape(const_tensor.dims().begin(),
                                  const_tensor.dims().end());
    while (tensor_shape.size() < 4) {
      tensor_shape.insert(tensor_shape.begin(), 1);
    }

    unsigned char *const_node_data = nullptr;
    int const_node_data_len = 0;
    if (!(const_tensor.data_type() == DataType::DT_INT32 &&
        const_tensor.data_size() == 0)) {
      const_node_data =
          const_cast<unsigned char *>(model_data + const_tensor.offset());
      const_node_data_len = const_tensor.data_size() *
          GetEnumTypeSize(const_tensor.data_type());
    }

    hexagon_hta_nn_append_const_node(nn_id_,
                                     node_id(const_tensor.node_id()),
                                     tensor_shape[0],
                                     tensor_shape[1],
                                     tensor_shape[2],
                                     tensor_shape[3],
                                     const_node_data,
                                     const_node_data_len);
  }

  // op node
  OpMap op_map;
  op_map.Init();
  std::vector<std::vector<hexagon_hta_nn_input>> cached_inputs;
  std::vector<std::vector<hexagon_hta_nn_output>> cached_outputs;
  std::vector<hexagon_hta_nn_input> inputs;
  std::vector<hexagon_hta_nn_output> outputs;

  for (const OperatorDef &op : net_def.op()) {
    hta_op_type op_id = op_map.GetOpId(op.type());
    inputs.resize(op.node_input().size());
    for (int i = 0; i < op.node_input().size(); ++i) {
      inputs[i].src_id = node_id(op.node_input()[i].node_id());
      inputs[i].output_idx = op.node_input()[i].output_port();
    }
    outputs.resize(op.output_shape().size());
    for (int i = 0; i < op.output_shape().size(); ++i) {
      outputs[i].rank = op.output_shape()[i].dims().size();
      for (size_t j = 0; j < outputs[i].rank; ++j) {
        outputs[i].max_sizes[j] = op.output_shape()[i].dims()[j];
      }
      if (outputs[i].rank == 0) {
        outputs[i].rank = 1;
        outputs[i].max_sizes[0] = 1;
      }
      outputs[i].max_sizes[outputs[i].rank] = 0;
      outputs[i].elementsize = GetEnumTypeSize(
          static_cast<DataType>(op.output_type()[i]));
      outputs[i].zero_offset = 0;
      outputs[i].stepsize = 0;
    }
    cached_inputs.push_back(inputs);
    cached_outputs.push_back(outputs);

    auto padding_type = static_cast<hta_padding_type>(op.padding());

    hexagon_hta_nn_append_node(nn_id_,
                               node_id(op.node_id()),
                               op_id,
                               padding_type,
                               cached_inputs.back().data(),
                               inputs.size(),
                               cached_outputs.back().data(),
                               outputs.size());
  }

  int64_t t1 = NowMicros();

  MACE_CHECK(hexagon_hta_nn_prepare(nn_id_) == 0, "hexagon_nn_prepare failed");

  int64_t t2 = NowMicros();

  VLOG(1) << "Setup time: " << t1 - t0 << " " << t2 - t1;

  // input info
  num_inputs_ = net_def.input_info_size();
  input_info_.reserve(num_inputs_);
  input_tensordef_.resize(num_inputs_);
  for (int index = 0; index < num_inputs_; ++index) {
    auto input_info = net_def.input_info(index);
    std::vector<index_t> input_shape(input_info.dims().begin(),
                                     input_info.dims().end());
    while (input_shape.size() < 4) {
      input_shape.insert(input_shape.begin(), 1);
    }

    auto quantized_tensor = make_unique<Tensor>(allocator_, DT_UINT8);
    auto hta_tensor = make_unique<Tensor>(allocator_, DT_UINT8);
    hexagon_hta_nn_hw_tensordef &input_tensordef = input_tensordef_[index];
    memset(&input_tensordef, 0, sizeof(input_tensordef));
    MACE_CHECK(hexagon_hta_nn_get_memory_layout(nn_id_, 0, index,
                                                &input_tensordef) == 0);
    input_tensordef.dataLen = input_tensordef.batchStride;
    VLOG(1) << input_tensordef.format << " " << input_tensordef.elementSize
            << " " << input_tensordef.numDims << " "
            << input_tensordef.batchStride;
    for (uint32_t i = 0; i < input_tensordef.numDims; ++i) {
      VLOG(1) << input_tensordef.dim[i].length << " "
              << input_tensordef.dim[i].lpadding << " "
              << input_tensordef.dim[i].valid;
    }
    hta_tensor->Resize({input_tensordef.dataLen});
    MACE_CHECK(hta_tensor->raw_size() == input_tensordef.dataLen);
    Tensor::MappingGuard input_guard(hta_tensor.get());
    input_tensordef.fd =
        allocator_->rpcmem()->ToFd(hta_tensor->mutable_data<void>());
    MACE_CHECK(hexagon_hta_nn_register_tensor(nn_id_, &input_tensordef) == 0);

    transformer_->SetInputTransformer(input_tensordef.format);

    input_info_.emplace_back(
        input_info.name(), input_shape, input_info.data_type(),
        input_info.scale(), input_info.zero_point(),
        std::move(quantized_tensor),
        std::move(hta_tensor));
  }

  // output info
  num_outputs_ = net_def.output_info_size();
  output_info_.reserve(num_outputs_);
  output_tensordef_.resize(num_outputs_);
  for (int index = 0; index < num_outputs_; ++index) {
    auto output_info = net_def.output_info(index);
    std::vector<index_t> output_shape(output_info.dims().begin(),
                                      output_info.dims().end());
    while (output_shape.size() < 4) {
      output_shape.insert(output_shape.begin(), 1);
    }

    auto quantized_tensor = make_unique<Tensor>(allocator_, DT_UINT8);
    auto hta_tensor = make_unique<Tensor>(allocator_, DT_UINT8);
    quantized_tensor->SetScale(output_info.scale());
    quantized_tensor->SetZeroPoint(output_info.zero_point());

    hexagon_hta_nn_hw_tensordef &output_tensordef = output_tensordef_[index];
    memset(&output_tensordef, 0, sizeof(output_tensordef));
    MACE_CHECK(hexagon_hta_nn_get_memory_layout(nn_id_, 1, index,
                                                &output_tensordef) == 0);
    output_tensordef.dataLen = output_tensordef.batchStride;
    VLOG(1) << output_tensordef.format << " " << output_tensordef.elementSize
            << " " << output_tensordef.numDims << " "
            << output_tensordef.batchStride;
    for (uint32_t i = 0; i < output_tensordef.numDims; ++i) {
      VLOG(1) << output_tensordef.dim[i].length << " "
              << output_tensordef.dim[i].lpadding << " "
              << output_tensordef.dim[i].valid;
    }
    hta_tensor->Resize({output_tensordef.batchStride});
    MACE_CHECK(hta_tensor->raw_size() == output_tensordef.dataLen);
    Tensor::MappingGuard output_guard(hta_tensor.get());
    output_tensordef.fd =
        allocator_->rpcmem()->ToFd(hta_tensor->mutable_data<void>());
    MACE_CHECK(hexagon_hta_nn_register_tensor(nn_id_, &output_tensordef) == 0);

    transformer_->SetOutputTransformer(output_tensordef.format);

    output_info_.emplace_back(
        output_info.name(), output_shape, output_info.data_type(),
        output_info.scale(), output_info.zero_point(),
        std::move(quantized_tensor), std::move(hta_tensor));

    VLOG(1) << "OutputInfo: "
            << "\n\t shape: " << output_shape[0] << " " << output_shape[1]
            << " " << output_shape[2] << " " << output_shape[3]
            << "\n\t type: " << output_info.data_type();
  }

  return true;
}

bool HexagonHTAWrapper::TeardownGraph() {
  LOG(INFO) << "Hexagon teardown graph";
  return hexagon_hta_nn_teardown(nn_id_) == 0;
}

void HexagonHTAWrapper::PrintLog() {
  LOG(INFO) << "Print Log";
}

void HexagonHTAWrapper::PrintGraph() {
  LOG(INFO) << "Print Graph";
}

void HexagonHTAWrapper::SetDebugLevel(int level) {
  LOG(INFO) << "Set debug level: " << level;
  MACE_CHECK(hexagon_hta_nn_set_debug_level(nn_id_, level) == 0,
             "set debug level error");
}

void HexagonHTAWrapper::GetPerfInfo() {
  LOG(INFO) << "Get perf info";
}

void HexagonHTAWrapper::ResetPerfInfo() {
  LOG(INFO) << "Reset perf info";
}

bool HexagonHTAWrapper::ExecuteGraph(const Tensor &input_tensor,
                                     Tensor *output_tensor) {
  MACE_UNUSED(input_tensor);
  MACE_UNUSED(output_tensor);
  MACE_NOT_IMPLEMENTED;
  return false;
}

bool HexagonHTAWrapper::ExecuteGraphNew(
    const std::map<std::string, Tensor *> &input_tensors,
    std::map<std::string, Tensor *> *output_tensors) {
  VLOG(2) << "Execute graph new: " << nn_id_;
  auto num_inputs = static_cast<uint32_t>(input_tensors.size());
  auto num_outputs = static_cast<uint32_t>(output_tensors->size());
  MACE_CHECK(num_inputs_ == static_cast<int>(num_inputs), "Wrong inputs num");
  MACE_CHECK(num_outputs_ == static_cast<int>(num_outputs),
             "Wrong outputs num");

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto input_tensor = input_tensors.at(input_info_[i].name);
    input_tensor->SetScale(input_info_[i].scale);
    input_tensor->SetZeroPoint(input_info_[i].zero_point);
    MACE_CHECK_SUCCESS(
        transformer_->Quantize(input_tensors.at(input_info_[i].name),
                               input_info_[i].quantized_tensor.get()));

    MACE_CHECK_SUCCESS(transformer_->TransformInput(
        input_info_[i].quantized_tensor.get(),
        input_info_[i].hta_tensor.get(), i));

    Tensor::MappingGuard input_guard(input_info_[i].hta_tensor.get());
  }

  MACE_CHECK(hexagon_hta_nn_execute_hw(nn_id_,
                                       input_tensordef_.data(), num_inputs,
                                       output_tensordef_.data(), num_outputs,
                                       nullptr, nullptr) == 0);

  for (size_t i = 0; i < num_outputs; ++i) {
    { // To sync cache
      Tensor::MappingGuard output_guard(output_info_[i].hta_tensor.get());
    }
    output_info_[i].quantized_tensor->Resize(output_info_[i].shape);
    transformer_->TransformOutput(output_info_[i].hta_tensor.get(),
                                  output_info_[i].quantized_tensor.get(), i);


    auto output_tensor = output_tensors->at(output_info_[i].name);
    MaceStatus st = transformer_->Dequantize(
        output_info_[i].quantized_tensor.get(), output_tensor);
  }

  return true;
}

}  // namespace mace
