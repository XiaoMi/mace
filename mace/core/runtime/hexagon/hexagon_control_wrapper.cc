// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#include <sys/time.h>
#include <thread>  // NOLINT(build/c++11)
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>

#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#include "mace/core/runtime/hexagon/hexagon_nn_ops.h"
#include "mace/core/types.h"

namespace {
inline int64_t NowMicros() {
  struct timeval tv;
  gettimeofday(&tv, nullptr);
  return static_cast<int64_t>(tv.tv_sec) * 1000000 + tv.tv_usec;
}
}

namespace mace {

#define MACE_MAX_NODE 2048

enum {
  NN_GRAPH_PERFEVENT_CYCLES = 0,
  NN_GRAPH_PERFEVENT_USER0 = 1,
  NN_GRAPH_PERFEVENT_USER1 = 2,
  NN_GRAPH_PERFEVENT_HWPMU = 3,
  NN_GRAPH_PERFEVENT_UTIME = 5,
};

int HexagonControlWrapper::GetVersion() {
  int version;
  MACE_CHECK(hexagon_nn_version(&version) == 0, "get version error");
  return version;
}

bool HexagonControlWrapper::Config() {
  LOG(INFO) << "Hexagon config";
  MACE_CHECK(hexagon_nn_set_powersave_level(0) == 0, "hexagon power error");
  MACE_CHECK(hexagon_nn_config() == 0, "hexagon config error");
  return true;
}

bool HexagonControlWrapper::Init() {
  LOG(INFO) << "Hexagon init";
#ifdef MACE_USE_NNLIB_OLD
  nn_id_ = hexagon_nn_init();
#else
  MACE_CHECK(hexagon_nn_init(&nn_id_) == 0, "hexagon_nn_init failed");
#endif
  ResetPerfInfo();
  return true;
}

bool HexagonControlWrapper::Finalize() {
  LOG(INFO) << "Hexagon finalize";
  return hexagon_nn_set_powersave_level(1) == 0;
}

bool HexagonControlWrapper::SetupGraph(const NetDef &net_def,
                                       unsigned const char *model_data) {
  LOG(INFO) << "Hexagon setup graph";

  int64_t t0 = NowMicros();

  // const node
#if defined(MACE_USE_NNLIB_CAF) || defined(MACE_USE_NNLIB_OLD)
  std::thread const_thread([&]()
#endif
  {
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
  }
#if defined(MACE_USE_NNLIB_CAF) || defined(MACE_USE_NNLIB_OLD)
  );  // NOLINT
#endif

  // op node
#if defined(MACE_USE_NNLIB_CAF) || defined(MACE_USE_NNLIB_OLD)
  std::thread op_thread([&]()
#endif
  {
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
#ifdef MACE_USE_NNLIB_OLD
        outputs[i].max_size = op.out_max_byte_size()[i];
#else
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
#endif
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
  }
#if defined(MACE_USE_NNLIB_CAF) || defined(MACE_USE_NNLIB_OLD)
  );  // NOLINT
  const_thread.join();
  op_thread.join();
#endif

  // input info
  num_inputs_ = 0;
  for (const InputInfo &input_info : net_def.input_info()) {
    std::vector<index_t> input_shape;
    input_shape.insert(input_shape.begin(), input_info.dims().begin(),
                       input_info.dims().end());
    while (input_shape.size() < 4) {
      input_shape.insert(input_shape.begin(), 1);
    }
    input_shapes_.push_back(input_shape);
    input_data_types_.push_back(input_info.data_type());
    num_inputs_ += 1;
  }

  // output info
  num_outputs_ = 0;
  for (const OutputInfo &output_info : net_def.output_info()) {
    std::vector<index_t> output_shape;
    output_shape.insert(output_shape.begin(), output_info.dims().begin(),
                        output_info.dims().end());
    while (output_shape.size() < 4) {
      output_shape.insert(output_shape.begin(), 1);
    }
    output_shapes_.push_back(output_shape);
    output_data_types_.push_back(output_info.data_type());
    num_outputs_ += 1;
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

bool HexagonControlWrapper::TeardownGraph() {
  LOG(INFO) << "Hexagon teardown graph";
  return hexagon_nn_teardown(nn_id_) == 0;
}

#define MACE_PRINT_BUFSIZE (2 * 1024 * 1024)

void HexagonControlWrapper::PrintLog() {
  char *buf;
  if ((buf = new char[MACE_PRINT_BUFSIZE]) == NULL) return;
  MACE_CHECK(hexagon_nn_getlog(nn_id_, reinterpret_cast<unsigned char *>(buf),
                               MACE_PRINT_BUFSIZE) == 0,
             "print log error");
  LOG(INFO) << std::string(buf);
  delete[] buf;
}

void HexagonControlWrapper::PrintGraph() {
  LOG(INFO) << "Print Graph";
  char *buf;
  if ((buf = new char[MACE_PRINT_BUFSIZE]) == NULL) return;
  MACE_CHECK(hexagon_nn_snpprint(nn_id_, reinterpret_cast<unsigned char *>(buf),
                                 MACE_PRINT_BUFSIZE) == 0,
             "print graph error");
  LOG(INFO) << std::string(buf);
  delete[] buf;
}

void HexagonControlWrapper::SetDebugLevel(int level) {
  LOG(INFO) << "Set debug level: " << level;
  MACE_CHECK(hexagon_nn_set_debug_level(nn_id_, level) == 0,
             "set debug level error");
}

void HexagonControlWrapper::GetPerfInfo() {
  LOG(INFO) << "Get perf info";
  std::vector<hexagon_nn_perfinfo> perf_info(MACE_MAX_NODE);
  unsigned int n_items = 0;
  MACE_CHECK(hexagon_nn_get_perfinfo(nn_id_, perf_info.data(), MACE_MAX_NODE,
                                     &n_items) == 0,
             "get perf info error");

  std::unordered_map<uint32_t, float> node_id_counters;
  std::unordered_map<std::string, std::pair<int, float>> node_type_counters;
  float total_duration = 0.0;

  VLOG(1) << "items: " << n_items;
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
    LOG(INFO) << "node id: " << perf_info[i].node_id
              << ", node type: " << node_type
              << ", node type id: " << node_type_id
              << ", executions: " << perf_info[i].executions
              << ", duration: " << node_id_counters[node_id];

    if (node_type_counters.find(node_type) == node_type_counters.end()) {
      node_type_counters[node_type] = {0, 0.0};
    }
    ++node_type_counters[node_type].first;
    node_type_counters[node_type].second += node_id_counters[node_id];
    total_duration += node_id_counters[node_id];
  }

  for (auto &node_type_counter : node_type_counters) {
    LOG(INFO) << "node type: " << node_type_counter.first
              << ", time: " << node_type_counter.second.first
              << ", duration: " << node_type_counter.second.second;
  }
  LOG(INFO) << "total duration: " << total_duration;
}

void HexagonControlWrapper::ResetPerfInfo() {
  LOG(INFO) << "Reset perf info";
  MACE_CHECK(hexagon_nn_reset_perfinfo(nn_id_, NN_GRAPH_PERFEVENT_UTIME) == 0,
             "reset perf error");
}

bool HexagonControlWrapper::ExecuteGraph(const Tensor &input_tensor,
                                         Tensor *output_tensor) {
  VLOG(2) << "Execute graph: " << nn_id_;
  // single input and single output
  MACE_ASSERT(num_inputs_ == 1, "Wrong inputs num");
  MACE_ASSERT(num_outputs_ == 1, "Wrong outputs num");
  output_tensor->SetDtype(output_data_types_[0]);
  output_tensor->Resize(output_shapes_[0]);
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

  MACE_ASSERT(output_shape.size() == output_shapes_[0].size(),
              "wrong output shape inferred");
  for (size_t i = 0; i < output_shape.size(); ++i) {
    MACE_ASSERT(static_cast<index_t>(output_shape[i]) == output_shapes_[0][i],
                "wrong output shape inferred");
  }
  MACE_ASSERT(output_bytes == output_tensor->raw_size(),
              "wrong output bytes inferred.");
  return res == 0;
}

bool HexagonControlWrapper::ExecuteGraphNew(
    const std::vector<Tensor> &input_tensors,
    std::vector<Tensor> *output_tensors) {
  LOG(INFO) << "Execute graph new: " << nn_id_;
  uint32_t num_inputs = static_cast<uint32_t>(input_tensors.size());
  uint32_t num_outputs = static_cast<uint32_t>(output_tensors->size());
  MACE_ASSERT(num_inputs_ == num_inputs, "Wrong inputs num");
  MACE_ASSERT(num_outputs_ == num_outputs, "Wrong outputs num");

  hexagon_nn_tensordef *inputs = new hexagon_nn_tensordef[num_inputs];
  hexagon_nn_tensordef *outputs = new hexagon_nn_tensordef[num_outputs];

  for (size_t i = 0; i < num_inputs; ++i) {
    std::vector<index_t> input_shape = input_tensors[i].shape();
    inputs[i].batches = static_cast<uint32_t>(input_shape[0]);
    inputs[i].height = static_cast<uint32_t>(input_shape[1]);
    inputs[i].width = static_cast<uint32_t>(input_shape[2]);
    inputs[i].depth = static_cast<uint32_t>(input_shape[3]);
    inputs[i].data = const_cast<unsigned char *>(
        reinterpret_cast<const unsigned char *>(input_tensors[i].raw_data()));
    inputs[i].dataLen = static_cast<int>(input_tensors[i].raw_size());
    inputs[i].data_valid_len = static_cast<uint32_t>(
        input_tensors[i].raw_size());
    inputs[i].unused = 0;
  }

  for (size_t i = 0; i < num_outputs; ++i) {
    (*output_tensors)[i].SetDtype(output_data_types_[i]);
    (*output_tensors)[i].Resize(output_shapes_[i]);
    outputs[i].data = reinterpret_cast<unsigned char *>(
        (*output_tensors)[i].raw_mutable_data());
    outputs[i].dataLen = static_cast<int>((*output_tensors)[i].raw_size());
  }

  int res =
      hexagon_nn_execute_new(nn_id_, inputs, num_inputs, outputs, num_outputs);

  for (size_t i = 0; i < num_outputs; ++i) {
    std::vector<uint32_t> output_shape{outputs[i].batches, outputs[i].height,
                                       outputs[i].width, outputs[i].depth};
    MACE_ASSERT(output_shape.size() == output_shapes_[i].size(),
                "wrong output shape inferred");
    for (size_t j = 0; j < output_shape.size(); ++j) {
      MACE_ASSERT(static_cast<index_t>(output_shape[j])
                      == output_shapes_[i][j],
                  "wrong output shape inferred");
    }
    MACE_ASSERT(static_cast<index_t>(outputs[i].data_valid_len)
                    == (*output_tensors)[i].raw_size(),
                "wrong output bytes inferred.");
  }

  delete[] inputs;
  delete[] outputs;
  return res == 0;
}

bool HexagonControlWrapper::ExecuteGraphPreQuantize(const Tensor &input_tensor,
                                                    Tensor *output_tensor) {
  std::vector<Tensor> input_tensors(3);
  std::vector<Tensor> output_tensors(3);
  input_tensors[0].SetDtype(DT_UINT8);
  output_tensors[0].SetDtype(DT_UINT8);
  input_tensors[0].ResizeLike(input_tensor);
  input_tensors[1].Resize({1, 1, 1, 1});
  float *min_in_data = input_tensors[1].mutable_data<float>();
  input_tensors[2].Resize({1, 1, 1, 1});
  float *max_in_data = input_tensors[2].mutable_data<float>();
  quantizer_.Quantize(input_tensor, &input_tensors[0], min_in_data,
                      max_in_data);
  if (!ExecuteGraphNew(input_tensors, &output_tensors)) {
    return false;
  }

  output_tensor->ResizeLike(output_tensors[0]);

  const float *min_out_data = output_tensors[1].data<float>();
  const float *max_out_data = output_tensors[2].data<float>();
  quantizer_.DeQuantize(output_tensors[0], *min_out_data, *max_out_data,
                        output_tensor);
  return true;
}

}  // namespace mace
