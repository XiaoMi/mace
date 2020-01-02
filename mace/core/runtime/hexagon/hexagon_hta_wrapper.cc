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

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <utility>

#include "mace/core/runtime/hexagon/hexagon_hta_ops.h"
#include "mace/core/types.h"
#include "mace/utils/memory.h"
#include "mace/core/quantize.h"
#include "third_party/hta/hta_hexagon_api.h"

namespace mace {

namespace {
struct InputOutputMetadata {
  void Init(float min_val, float max_val, int needs_quantization) {
    this->min_val = min_val;
    this->max_val = max_val;
    this->needs_quantization = needs_quantization;
  }
  float min_val;
  float max_val;
  int needs_quantization;
};

template<typename T>
void AddInputMetadata(const T &data, hexagon_hta_nn_tensordef *tensor) {
  tensor->batches = 1;
  tensor->height = 1;
  tensor->width = 1;
  tensor->depth = 1;
  tensor->data = const_cast<unsigned char *>(
      reinterpret_cast<const unsigned char *>(&data));
  tensor->dataLen = sizeof(data);
  tensor->data_valid_len = sizeof(data);
  tensor->unused = 0;
}

template<typename T>
void AddOutputMetadata(const T &data, hexagon_hta_nn_tensordef *tensor) {
  tensor->data = const_cast<unsigned char *>(
      reinterpret_cast<const unsigned char *>(&data));
  tensor->dataLen = sizeof(data);
}
}  // namespace

HexagonHTAWrapper::HexagonHTAWrapper(Device *device)
    : quantize_util_(&device->cpu_runtime()->thread_pool()) {
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

  // input info
  num_inputs_ = net_def.input_info_size();
  input_info_.reserve(num_inputs_);
  for (const InputOutputInfo &input_info : net_def.input_info()) {
    std::vector<index_t> input_shape(input_info.dims().begin(),
                                     input_info.dims().end());
    while (input_shape.size() < 4) {
      input_shape.insert(input_shape.begin(), 1);
    }
    input_info_.emplace_back(input_info.name(),
                             input_shape,
                             input_info.data_type(),
                             input_info.scale(),
                             input_info.zero_point(),
                             make_unique<Tensor>());
  }

  // output info
  num_outputs_ = net_def.output_info_size();
  output_info_.reserve(num_outputs_);
  for (const InputOutputInfo &output_info : net_def.output_info()) {
    std::vector<index_t> output_shape(output_info.dims().begin(),
                                      output_info.dims().end());
    while (output_shape.size() < 4) {
      output_shape.insert(output_shape.begin(), 1);
    }
    output_info_.emplace_back(output_info.name(),
                              output_shape,
                              output_info.data_type(),
                              output_info.scale(),
                              output_info.zero_point(),
                              make_unique<Tensor>());
    VLOG(1) << "OutputInfo: "
            << "\n\t shape: " << output_shape[0] << " " << output_shape[1]
            << " " << output_shape[2] << " " << output_shape[3]
            << "\n\t type: " << output_info.data_type();
  }

  int64_t t1 = NowMicros();

  MACE_CHECK(hexagon_hta_nn_prepare(nn_id_) == 0, "hexagon_nn_prepare failed");

  int64_t t2 = NowMicros();

  VLOG(1) << "Setup time: " << t1 - t0 << " " << t2 - t1;

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

  std::vector<hexagon_hta_nn_tensordef> inputs(num_inputs * kNumMetaData);
  std::vector<hexagon_hta_nn_tensordef> outputs(num_outputs * kNumMetaData);
  std::vector<InputOutputMetadata> input_metadata(num_inputs);
  std::vector<InputOutputMetadata> output_metadata(num_outputs);

  for (size_t i = 0; i < num_inputs; ++i) {
    const auto input_tensor = input_tensors.at(input_info_[i].name);
    const auto &input_shape = input_tensor->shape();
    size_t index = i * kNumMetaData;
    inputs[index].batches = static_cast<uint32_t>(input_shape[0]);
    inputs[index].height = static_cast<uint32_t>(input_shape[1]);
    inputs[index].width = static_cast<uint32_t>(input_shape[2]);
    inputs[index].depth = static_cast<uint32_t>(input_shape[3]);
    inputs[index].data = const_cast<unsigned char *>(
        reinterpret_cast<const unsigned char *>(input_tensor->raw_data()));
    inputs[index].dataLen = static_cast<int>(input_tensor->raw_size());
    inputs[index].data_valid_len =
        static_cast<uint32_t>(input_tensor->raw_size());
    inputs[index].unused = 0;
    input_metadata[i].Init(.0f, .0f, 1);
    AddInputMetadata(input_metadata[i].min_val, &inputs[index + 1]);
    AddInputMetadata(input_metadata[i].max_val, &inputs[index + 2]);
    AddInputMetadata(input_metadata[i].needs_quantization, &inputs[index + 3]);
  }

  // transform mace output to hexagon output
  for (size_t i = 0; i < num_outputs; ++i) {
    auto output_tensor = output_tensors->at(output_info_[i].name);
    size_t index = i * kNumMetaData;
    output_tensor->SetDtype(output_info_[i].data_type);
    output_tensor->Resize(output_info_[i].shape);

    outputs[index].data = reinterpret_cast<unsigned char *>(
        output_tensor->raw_mutable_data());
    outputs[index].dataLen = static_cast<int>(output_tensor->raw_size());
    output_metadata[i].Init(.0f, .0f, 1);

    AddOutputMetadata(output_metadata[i].min_val, &outputs[index + 1]);
    AddOutputMetadata(output_metadata[i].max_val, &outputs[index + 2]);
    AddOutputMetadata(output_metadata[i].needs_quantization,
                      &outputs[index + 3]);
  }

  int res = hexagon_hta_nn_execute_new(nn_id_,
                                       inputs.data(),
                                       num_inputs * kNumMetaData,
                                       outputs.data(),
                                       num_outputs * kNumMetaData);
  MACE_CHECK(res == 0, "execute error");

  for (size_t i = 0; i < num_outputs; ++i) {
    size_t index = i * kNumMetaData;
    std::vector<uint32_t> output_shape{
        outputs[index].batches, outputs[index].height, outputs[index].width,
        outputs[index].depth};
    MACE_CHECK(output_shape.size() == output_info_[i].shape.size(),
               output_shape.size(), " vs ", output_info_[i].shape.size(),
               " wrong output shape inferred");
    for (size_t j = 0; j < output_shape.size(); ++j) {
      MACE_CHECK(static_cast<index_t>(output_shape[j])
                     == output_info_[i].shape[j],
                 output_shape[j], " vs ", output_info_[i].shape[j],
                 " wrong output shape[", j, "] inferred");
    }
    auto output_tensor = output_tensors->at(output_info_[i].name);
    MACE_CHECK(static_cast<index_t>(outputs[index].data_valid_len)
                   == output_tensor->raw_size(),
               outputs[index].data_valid_len, " vs ", output_tensor->raw_size(),
               " wrong output bytes inferred.");
  }

  return res == 0;
}

}  // namespace mace
