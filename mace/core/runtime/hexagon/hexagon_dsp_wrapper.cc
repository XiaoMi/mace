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

#include "mace/core/runtime/hexagon/hexagon_dsp_wrapper.h"

#include <algorithm>
#include <iomanip>
#include <map>
#include <memory>
#include <thread>  // NOLINT(build/c++11)
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

#include "mace/core/runtime/hexagon/hexagon_dsp_ops.h"
#include "mace/core/types.h"
#include "mace/port/env.h"
#include "mace/utils/memory.h"
#include "third_party/nnlib/hexagon_nn.h"

namespace mace {

#define MACE_MAX_NODE 2048

enum {
  NN_GRAPH_PERFEVENT_CYCLES = 0,
  NN_GRAPH_PERFEVENT_USER0 = 1,
  NN_GRAPH_PERFEVENT_USER1 = 2,
  NN_GRAPH_PERFEVENT_HWPMU = 3,
  NN_GRAPH_PERFEVENT_UTIME = 5,
};

namespace {
struct InputOutputMetadata{
  void Init(float min_val, float max_val, int needs_quantization) {
    this->min_val = min_val;
    this->max_val = max_val;
    this->needs_quantization = needs_quantization;
  }
  float min_val;
  float max_val;
  int needs_quantization;
};

template <typename T>
void AddInputMetadata(const T &data, hexagon_nn_tensordef *tensor) {
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

template <typename T>
void AddOutputMetadata(const T &data, hexagon_nn_tensordef *tensor) {
  tensor->data = const_cast<unsigned char *>(
      reinterpret_cast<const unsigned char *>(&data));
  tensor->dataLen = sizeof(data);
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
}  // namespace

int HexagonDSPWrapper::GetVersion() {
  int version;
  MACE_CHECK(hexagon_nn_version(&version) == 0, "get version error");
  return version;
}

bool HexagonDSPWrapper::Config() {
  LOG(INFO) << "Hexagon config";
  MACE_CHECK(hexagon_nn_set_powersave_level(0) == 0, "hexagon power error");
  MACE_CHECK(hexagon_nn_config() == 0, "hexagon config error");
  return true;
}

bool HexagonDSPWrapper::Init() {
  LOG(INFO) << "Hexagon init";
  MACE_CHECK(hexagon_nn_init(&nn_id_) == 0, "hexagon_nn_init failed");
  ResetPerfInfo();
  return true;
}

bool HexagonDSPWrapper::Finalize() {
  LOG(INFO) << "Hexagon finalize";
  return hexagon_nn_set_powersave_level(1) == 0;
}

bool HexagonDSPWrapper::SetupGraph(const NetDef &net_def,
                                   unsigned const char *model_data) {
  LOG(INFO) << "Hexagon setup graph";

  int64_t t0 = NowMicros();

  // const node
  std::vector<hexagon_nn_const_node> const_node_list;
  for (const ConstTensor &const_tensor : net_def.tensors()) {
    std::vector<int> tensor_shape(const_tensor.dims().begin(),
                                  const_tensor.dims().end());
    while (tensor_shape.size() < 4) {
      tensor_shape.insert(tensor_shape.begin(), 1);
    }

    hexagon_nn_const_node const_node;
    const_node.node_id = node_id(const_tensor.node_id());
    const_node.tensor.batches = tensor_shape[0];
    const_node.tensor.height = tensor_shape[1];
    const_node.tensor.width = tensor_shape[2];
    const_node.tensor.depth = tensor_shape[3];

    if (const_tensor.data_type() == DataType::DT_INT32 &&
        const_tensor.data_size() == 0) {
      const_node.tensor.data = NULL;
      const_node.tensor.dataLen = 0;
    } else {
      const_node.tensor.data =
          const_cast<unsigned char *>(model_data + const_tensor.offset());
      const_node.tensor.dataLen = const_tensor.data_size() *
                                  GetEnumTypeSize(const_tensor.data_type());
    }
    const_node_list.push_back(const_node);
    // 255 is magic number: why fastrpc limits sequence length to that?
    if (const_node_list.size() >= 250) {
      MACE_CHECK(
          hexagon_nn_append_const_node_list(nn_id_, const_node_list.data(),
                                            const_node_list.size()) == 0,
          "append const node error");
      const_node_list.clear();
    }
  }

  if (!const_node_list.empty()) {
    MACE_CHECK(
        hexagon_nn_append_const_node_list(nn_id_, const_node_list.data(),
                                          const_node_list.size()) == 0,
        "append const node error");
  }
  const_node_list.clear();

  // op node
  OpMap op_map;
  op_map.Init();
  std::vector<hexagon_nn_op_node> op_node_list;
  std::vector<std::vector<hexagon_nn_input>> cached_inputs;
  std::vector<std::vector<hexagon_nn_output>> cached_outputs;
  std::vector<hexagon_nn_input> inputs;
  std::vector<hexagon_nn_output> outputs;

  for (const OperatorDef &op : net_def.op()) {
    int op_id = op_map.GetOpId(op.type());
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

    hexagon_nn_padding_type padding_type =
        static_cast<hexagon_nn_padding_type>(op.padding());

    hexagon_nn_op_node op_node;
    op_node.node_id = node_id(op.node_id());
    op_node.operation = op_id;
    op_node.padding = padding_type;
    op_node.inputs = cached_inputs.back().data();
    op_node.inputsLen = inputs.size();
    op_node.outputs = cached_outputs.back().data();
    op_node.outputsLen = outputs.size();

    op_node_list.push_back(op_node);
    if (op_node_list.size() >= 125) {
      MACE_CHECK(hexagon_nn_append_node_list(nn_id_, op_node_list.data(),
                                             op_node_list.size()) == 0,
                 "append node error");
      op_node_list.clear();
      cached_inputs.clear();
      cached_outputs.clear();
    }
  }

  if (!op_node_list.empty()) {
    MACE_CHECK(hexagon_nn_append_node_list(nn_id_, op_node_list.data(),
                                           op_node_list.size()) == 0,
               "append node error");
  }
  op_node_list.clear();
  cached_inputs.clear();
  cached_outputs.clear();

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

  MACE_CHECK(hexagon_nn_prepare(nn_id_) == 0, "hexagon_nn_prepare failed");

  int64_t t2 = NowMicros();

  VLOG(1) << "Setup time: " << t1 - t0 << " " << t2 - t1;

  return true;
}

bool HexagonDSPWrapper::TeardownGraph() {
  LOG(INFO) << "Hexagon teardown graph";
  return hexagon_nn_teardown(nn_id_) == 0;
}

#define MACE_PRINT_BUFSIZE (2 * 1024 * 1024)

void HexagonDSPWrapper::PrintLog() {
  char *buf;
  if ((buf = new char[MACE_PRINT_BUFSIZE]) == NULL) return;
  MACE_CHECK(hexagon_nn_getlog(nn_id_, reinterpret_cast<unsigned char *>(buf),
                               MACE_PRINT_BUFSIZE) == 0,
             "print log error");
  LOG(INFO) << std::string(buf);
  delete[] buf;
}

void HexagonDSPWrapper::PrintGraph() {
  LOG(INFO) << "Print Graph";
  char *buf;
  if ((buf = new char[MACE_PRINT_BUFSIZE]) == NULL) return;
  MACE_CHECK(hexagon_nn_snpprint(nn_id_, reinterpret_cast<unsigned char *>(buf),
                                 MACE_PRINT_BUFSIZE) == 0,
             "print graph error");
  LOG(INFO) << std::string(buf);
  delete[] buf;
}

void HexagonDSPWrapper::SetDebugLevel(int level) {
  LOG(INFO) << "Set debug level: " << level;
  MACE_CHECK(hexagon_nn_set_debug_level(nn_id_, level) == 0,
             "set debug level error");
}

void HexagonDSPWrapper::GetPerfInfo() {
  LOG(INFO) << "Get perf info";
  std::vector<hexagon_nn_perfinfo> perf_info(MACE_MAX_NODE);
  unsigned int n_items = 0;
  MACE_CHECK(hexagon_nn_get_perfinfo(nn_id_, perf_info.data(), MACE_MAX_NODE,
                                     &n_items) == 0,
             "get perf info error");

  std::unordered_map<uint32_t, float> node_id_counters;
  std::unordered_map<std::string, std::pair<int, float>> node_type_counters;
  std::vector<std::string> node_types;
  float total_duration = 0.0;

  VLOG(1) << "items: " << n_items;
  std::string run_order_title = "Sort by Run Order";
  const std::vector<std::string> run_order_header = {
      "Node Id", "Node Type", "Node Type Id", "Executions", "Duration(ms)"
  };
  std::vector<std::vector<std::string>> run_order_data;
  for (unsigned int i = 0; i < n_items; ++i) {
    unsigned int node_id = perf_info[i].node_id;
    unsigned int node_type_id = perf_info[i].node_type;
    node_id_counters[node_id] =
        ((static_cast<uint64_t>(perf_info[i].counter_hi) << 32) +
         perf_info[i].counter_lo) *
        1.0f / perf_info[i].executions;

    char node_type_buf[MACE_MAX_NODE];
    hexagon_nn_op_id_to_name(node_type_id, node_type_buf, MACE_MAX_NODE);
    std::string node_type(node_type_buf);
    if (node_type.compare("Const") == 0) continue;
    std::vector<std::string> tuple;
    tuple.push_back(IntToString(perf_info[i].node_id));
    tuple.push_back(node_type);
    tuple.push_back(IntToString(node_type_id));
    tuple.push_back(IntToString(perf_info[i].executions));
    tuple.push_back(FloatToString(node_id_counters[node_id] / 1000.0f, 3));
    run_order_data.emplace_back(tuple);

    if (node_type_counters.find(node_type) == node_type_counters.end()) {
      node_type_counters[node_type] = {0, 0.0};
      node_types.push_back(node_type);
    }
    ++node_type_counters[node_type].first;
    node_type_counters[node_type].second += node_id_counters[node_id];
    total_duration += node_id_counters[node_id];
  }

  std::sort(node_types.begin(), node_types.end(),
            [&](const std::string &lhs, const std::string &rhs) {
              return node_type_counters[lhs].second
                  > node_type_counters[rhs].second;
            });

  std::string duration_title = "Sort by Duration";
  const std::vector<std::string> duration_header = {
      "Node Type", "Times", "Duration(ms)"
  };
  std::vector<std::vector<std::string>> duration_data;
  for (auto &node_type : node_types) {
    auto node_type_counter = node_type_counters[node_type];
    std::vector<std::string> tuple;
    tuple.push_back(node_type);
    tuple.push_back(IntToString(node_type_counter.first));
    tuple.push_back(FloatToString(node_type_counter.second / 1000.0f, 3));
    duration_data.emplace_back(tuple);
  }

  LOG(INFO) << mace::string_util::StringFormatter::Table(
      run_order_title, run_order_header, run_order_data);
  LOG(INFO) << mace::string_util::StringFormatter::Table(
      duration_title, duration_header, duration_data);
  LOG(INFO) << "total duration: " << std::fixed << total_duration;
}

void HexagonDSPWrapper::ResetPerfInfo() {
  LOG(INFO) << "Reset perf info";
  MACE_CHECK(hexagon_nn_reset_perfinfo(nn_id_, NN_GRAPH_PERFEVENT_UTIME) == 0,
             "reset perf error");
}

bool HexagonDSPWrapper::ExecuteGraph(const Tensor &input_tensor,
                                     Tensor *output_tensor) {
  VLOG(2) << "Execute graph: " << nn_id_;
  // single input and single output
  MACE_CHECK(num_inputs_ == 1, "Wrong inputs num");
  MACE_CHECK(num_outputs_ == 1, "Wrong outputs num");
  output_tensor->SetDtype(output_info_[0].data_type);
  output_tensor->Resize(output_info_[0].shape);
  std::vector<uint32_t> output_shape(4);
  uint32_t output_bytes;
  int res = hexagon_nn_execute(
      nn_id_,
      static_cast<uint32_t>(input_tensor.shape()[0]),
      static_cast<uint32_t>(input_tensor.shape()[1]),
      static_cast<uint32_t>(input_tensor.shape()[2]),
      static_cast<uint32_t>(input_tensor.shape()[3]),
      reinterpret_cast<const unsigned char *>(input_tensor.raw_data()),
      static_cast<int>(input_tensor.raw_size()),
      &output_shape[0],
      &output_shape[1],
      &output_shape[2],
      &output_shape[3],
      reinterpret_cast<unsigned char *>(output_tensor->raw_mutable_data()),
      static_cast<int>(output_tensor->raw_size()),
      &output_bytes);
  MACE_CHECK(res == 0, "execute error");

  MACE_CHECK(output_shape.size() == output_info_[0].shape.size(),
             "wrong output shape inferred");
  for (size_t i = 0; i < output_shape.size(); ++i) {
    MACE_CHECK(static_cast<index_t>(output_shape[i])
                   == output_info_[0].shape[i],
               "wrong output shape inferred");
  }
  MACE_CHECK(output_bytes == output_tensor->raw_size(),
             "wrong output bytes inferred.");
  return res == 0;
}

bool HexagonDSPWrapper::ExecuteGraphNew(
    const std::map<std::string, Tensor*> &input_tensors,
    std::map<std::string, Tensor*> *output_tensors) {
  VLOG(2) << "Execute graph new: " << nn_id_;
  uint32_t num_inputs = static_cast<uint32_t>(input_tensors.size());
  uint32_t num_outputs = static_cast<uint32_t>(output_tensors->size());
  MACE_CHECK(num_inputs_ == static_cast<int>(num_inputs), "Wrong inputs num");
  MACE_CHECK(num_outputs_ == static_cast<int>(num_outputs),
             "Wrong outputs num");

  std::vector<hexagon_nn_tensordef> inputs(num_inputs * kNumMetaData);
  std::vector<hexagon_nn_tensordef> outputs(num_outputs * kNumMetaData);
  std::vector<InputOutputMetadata> input_metadata(num_inputs);
  std::vector<InputOutputMetadata> output_metadata(num_outputs);

  // transform mace input to hexagon input
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

  // Execute graph
  int res = hexagon_nn_execute_new(nn_id_,
                                   inputs.data(),
                                   num_inputs * kNumMetaData,
                                   outputs.data(),
                                   num_outputs * kNumMetaData);

  // handle hexagon output
  for (size_t i = 0; i < num_outputs; ++i) {
    size_t index = i * kNumMetaData;
    std::vector<uint32_t> output_shape{
        outputs[index].batches, outputs[index].height, outputs[index].width,
        outputs[index].depth};
    MACE_CHECK(output_shape.size() == output_info_[i].shape.size(),
               output_shape.size(), " vs ", output_info_[i].shape.size(),
                "wrong output shape inferred");
    for (size_t j = 0; j < output_shape.size(); ++j) {
      MACE_CHECK(static_cast<index_t>(output_shape[j])
                     == output_info_[i].shape[j],
                 output_shape[j], " vs ", output_info_[i].shape[j],
                 "wrong output shape inferred");
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
