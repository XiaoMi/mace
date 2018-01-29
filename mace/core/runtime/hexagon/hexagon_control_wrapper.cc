//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/core/runtime/hexagon/hexagon_control_wrapper.h"
#include <fstream>
#include <unordered_map>

namespace mace {

#define MAX_NODE 2048

enum {
  NN_GRAPH_PERFEVENT_CYCLES = 0,
  NN_GRAPH_PERFEVENT_USER0 = 1,
  NN_GRAPH_PERFEVENT_USER1 = 2,
  NN_GRAPH_PERFEVENT_HWPMU = 3,
  NN_GRAPH_PERFEVENT_UTIME = 5,
};

int HexagonControlWrapper::GetVersion() {
  int version;
  hexagon_nn_version(&version);
  return version;
}

bool HexagonControlWrapper::Config() {
  LOG(INFO) << "Hexagon config";
  if (hexagon_controller_InitHexagonWithMaxAttributes(0, 100) != 0) {
    return false;
  }
  return hexagon_nn_config() == 0;
}

bool HexagonControlWrapper::Init() {
  LOG(INFO) << "Hexagon init";
  nn_id_ = hexagon_nn_init();
  ResetPerfInfo();
  return nn_id_ != 0;
}

bool HexagonControlWrapper::Finalize() {
  LOG(INFO) << "Hexagon finalize";
  return hexagon_controller_DeInitHexagon() == 0;
}

bool HexagonControlWrapper::SetupGraph(const NetDef& net_def) {
  LOG(INFO) << "Hexagon setup graph";
  // const node
  for (const ConstTensor& tensor_proto: net_def.tensors()) {
    vector<int> tensor_shape(tensor_proto.dims().begin(),
                             tensor_proto.dims().end());
    while (tensor_shape.size() < 4) {
      tensor_shape.insert(tensor_shape.begin(), 1);
    }

    if (tensor_proto.data_type() == DataType::DT_INT32
      && tensor_proto.data_size() == 0) {
      hexagon_nn_append_const_node(nn_id_, node_id(tensor_proto.node_id()),
                                   tensor_shape[0], tensor_shape[1],
                                   tensor_shape[2], tensor_shape[3],
                                   NULL,
                                   0);
    } else {
      unique_ptr<Tensor> tensor = serializer_.Deserialize(tensor_proto,
                                                          DeviceType::CPU);
      hexagon_nn_append_const_node(nn_id_, node_id(tensor_proto.node_id()),
                                   tensor_shape[0], tensor_shape[1],
                                   tensor_shape[2], tensor_shape[3],
                                   reinterpret_cast<const unsigned char *>(
                                     tensor->raw_data()),
                                   tensor->raw_size());
    }
    VLOG(1) << "Const: " << tensor_proto.name()
            << ", node_id: " << node_id(tensor_proto.node_id())
            << "\n\t shape: " << tensor_shape[0] << " " << tensor_shape[1]
            << " " << tensor_shape[2] << " " << tensor_shape[3];
  }

  // op node
  for (const OperatorDef& op: net_def.op()) {
    unsigned int op_id;
    MACE_CHECK(hexagon_nn_op_name_to_id(op.type().data(), &op_id) == 0,
                "invalid op: ", op.name(), ", type: ", op.type());
    vector<hexagon_nn_input> inputs(op.node_input().size());
    for (size_t i = 0; i < op.node_input().size(); ++i) {
      inputs[i].src_id = node_id(op.node_input()[i].node_id());
      inputs[i].output_idx = op.node_input()[i].output_port();
    }
    vector<hexagon_nn_output> outputs(op.out_max_byte_size().size());
    for (size_t i = 0; i < op.out_max_byte_size().size(); ++i) {
      outputs[i].max_size = op.out_max_byte_size()[i];
    }

    hexagon_nn_padding_type padding_type = static_cast<hexagon_nn_padding_type>(
      op.padding());

    hexagon_nn_append_node(nn_id_, node_id(op.node_id()), op_id, padding_type,
                           inputs.data(), inputs.size(),
                           outputs.data(), outputs.size());

    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Op: " << op.name()
              << ", type: " << op.type()
              << ", node_id: " << node_id(op.node_id())
              << ", padding_type: " << padding_type;

      for (const auto &input: inputs) {
        VLOG(1) << "\t input: " << input.src_id << ":" << input.output_idx;
      }
      for (const auto &output: outputs) {
        VLOG(1) << "\t output: " << output.max_size;
      }
    }
  }

  // input info
  num_inputs_ = 0;
  for (const InputInfo &input_info: net_def.input_info()) {
    vector<index_t> input_shape;
    input_shape.insert(input_shape.begin(),
                       input_info.dims().begin(), input_info.dims().end());
    while (input_shape.size() < 4) {
      input_shape.insert(input_shape.begin(), 1);
    }
    input_shapes_.push_back(input_shape);
    input_data_types_.push_back(input_info.data_type());
    num_inputs_ += 1;
  }

  // output info
  num_outputs_ = 0;
  for (const OutputInfo &output_info: net_def.output_info()) {
    vector<index_t> output_shape;
    output_shape.insert(output_shape.begin(),
                        output_info.dims().begin(), output_info.dims().end());
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

  return hexagon_nn_prepare(nn_id_) == 0;
}

bool HexagonControlWrapper::TeardownGraph() {
  LOG(INFO) << "Hexagon teardown graph";
  return hexagon_nn_teardown(nn_id_) == 0;
}

#define PRINT_BUFSIZE (2*1024*1024)

void HexagonControlWrapper::PrintLog() {
  char *buf;
  unsigned char *p;
  if ((buf = new char[PRINT_BUFSIZE]) == NULL) return;
  hexagon_nn_getlog(nn_id_, reinterpret_cast<unsigned char*>(buf), PRINT_BUFSIZE);
  LOG(INFO) << string(buf);
  delete []buf;
}

void HexagonControlWrapper::PrintGraph() {
  LOG(INFO) << "Print Graph";
  char *buf;
  unsigned char *p;
  if ((buf = new char[PRINT_BUFSIZE]) == NULL) return;
  hexagon_nn_snpprint(nn_id_, reinterpret_cast<unsigned char*>(buf), PRINT_BUFSIZE);
  LOG(INFO) << string(buf);
  delete []buf;
}

void HexagonControlWrapper::SetDebugLevel(int level) {
  LOG(INFO) << "Set debug level: " << level;
  hexagon_nn_set_debug_level(nn_id_, level);
}

void HexagonControlWrapper::SetGraphMode(int mode) {
  LOG(INFO) << "Set dsp mode: " << mode;
  hexagon_nn_set_graph_mode(nn_id_, mode);
}

void HexagonControlWrapper::GetPerfInfo() {
  LOG(INFO) << "Get perf info";
  vector<hexagon_nn_perfinfo> perf_info(MAX_NODE);
  unsigned int n_items = 0;
  hexagon_nn_get_perfinfo(nn_id_, perf_info.data(), MAX_NODE, &n_items);

  std::unordered_map<uint32_t, float> node_id_counters;
  std::unordered_map<std::string, std::pair<int, float>> node_type_counters;
  float total_duration = 0.0;

  VLOG(0) << "items: " << n_items;
  for (int i = 0; i < n_items; ++i) {
    unsigned int node_id = perf_info[i].node_id;
    unsigned int node_type_id = perf_info[i].node_type;
    node_id_counters[node_id] = ((static_cast<uint64_t>(perf_info[i].counter_hi) << 32)
      + perf_info[i].counter_lo) * 1.0f / perf_info[i].executions;

    char node_type_buf[MAX_NODE];
    hexagon_nn_op_id_to_name(node_type_id, node_type_buf, MAX_NODE);
    std::string node_type(node_type_buf);
    LOG(INFO) << "node id: " << perf_info[i].node_id
              << ", node type: " << node_type
              << ", executions: " << perf_info[i].executions
              << ", duration: " << node_id_counters[node_id];

    if (node_type_counters.find(node_type) == node_type_counters.end()) {
      node_type_counters[node_type] = {0, 0.0};
    }
    ++node_type_counters[node_type].first;
    node_type_counters[node_type].second += node_id_counters[node_id];
    total_duration += node_id_counters[node_id];
  }

  for (auto& node_type_counter: node_type_counters) {
    LOG(INFO) << "node type: " << node_type_counter.first
              << ", time: " << node_type_counter.second.first
              << ", duration: " << node_type_counter.second.second;
  }
  LOG(INFO) << "total duration: " << total_duration;
}

void HexagonControlWrapper::ResetPerfInfo() {
  LOG(INFO) << "Reset perf info";
  hexagon_nn_reset_perfinfo(nn_id_, NN_GRAPH_PERFEVENT_UTIME);
}

bool HexagonControlWrapper::ExecuteGraph(const Tensor &input_tensor,
                                         Tensor *output_tensor) {
  LOG(INFO) << "Execute graph: " << nn_id_;
  // single input and single output
  MACE_ASSERT(num_inputs_ == 1, "Wrong inputs num");
  MACE_ASSERT(num_outputs_ == 1, "Wrong outputs num");
  output_tensor->SetDtype(output_data_types_[0]);
  output_tensor->Resize(output_shapes_[0]);
  vector<uint32_t> output_shape(4);
  uint32_t output_bytes;
  int res = hexagon_nn_execute(nn_id_,
                               input_tensor.shape()[0],
                               input_tensor.shape()[1],
                               input_tensor.shape()[2],
                               input_tensor.shape()[3],
                               reinterpret_cast<const unsigned char *>(
                                 input_tensor.raw_data()),
                               input_tensor.raw_size(),
                               &output_shape[0],
                               &output_shape[1],
                               &output_shape[2],
                               &output_shape[3],
                               reinterpret_cast<unsigned char *>(
                                 output_tensor->raw_mutable_data()),
                               output_tensor->raw_size(),
                               &output_bytes);

  MACE_ASSERT(output_shape == output_shapes_[0],
              "wrong output shape inferred");
  MACE_ASSERT(output_bytes == output_tensor->raw_size(),
              "wrong output bytes inferred.");
  return res == 0;
};

bool HexagonControlWrapper::ExecuteGraphNew(const vector<Tensor> &input_tensors,
                                            vector<Tensor> *output_tensors) {
  LOG(INFO) << "Execute graph new: " << nn_id_;
  int num_inputs = input_tensors.size();
  int num_outputs = output_tensors->size();
  MACE_ASSERT(num_inputs_ == num_inputs, "Wrong inputs num");
  MACE_ASSERT(num_outputs_ == num_outputs, "Wrong outputs num");

  hexagon_nn_tensordef *inputs = new hexagon_nn_tensordef[num_inputs];
  hexagon_nn_tensordef *outputs = new hexagon_nn_tensordef[num_outputs];

  for (int i = 0; i < num_inputs; ++i) {
    vector<index_t> input_shape = input_tensors[i].shape();
    inputs[i].batches = input_shape[0];
    inputs[i].height = input_shape[1];
    inputs[i].width = input_shape[2];
    inputs[i].depth = input_shape[3];
    inputs[i].data = const_cast<unsigned char *>(
      reinterpret_cast<const unsigned char *>(input_tensors[i].raw_data()));
    inputs[i].dataLen = input_tensors[i].raw_size();
    inputs[i].data_valid_len = input_tensors[i].raw_size();
    inputs[i].unused = 0;
  }

  for (int i = 0; i < num_outputs; ++i) {
    (*output_tensors)[i].SetDtype(output_data_types_[i]);
    (*output_tensors)[i].Resize(output_shapes_[i]);
    outputs[i].data = reinterpret_cast<unsigned char *>(
      (*output_tensors)[i].raw_mutable_data());
    outputs[i].dataLen = (*output_tensors)[i].raw_size();
  }

  int res = hexagon_nn_execute_new(nn_id_, inputs, num_inputs,
                                   outputs, num_outputs);

  for (int i = 0; i < num_outputs; ++i) {
    vector<uint32_t> output_shape {outputs[i].batches, outputs[i].height,
                                   outputs[i].width, outputs[i].depth};
    MACE_ASSERT(output_shape  == output_shapes_[i],
                "wrong output shape inferred");
    MACE_ASSERT(outputs[i].data_valid_len == (*output_tensors)[i].raw_size(),
                "wrong output bytes inferred.");
  }

  delete [] inputs;
  delete [] outputs;
  return res == 0;
};

bool HexagonControlWrapper::ExecuteGraphPreQuantize(const Tensor &input_tensor,
                                                    Tensor *output_tensor) {
  vector<Tensor> input_tensors(3);
  vector<Tensor> output_tensors(3);
  input_tensors[0].SetDtype(DT_UINT8);
  output_tensors[0].SetDtype(DT_UINT8);
  input_tensors[0].ResizeLike(input_tensor);
  input_tensors[1].Resize({1, 1, 1, 1});
  float *min_in_data = input_tensors[1].mutable_data<float>();
  input_tensors[2].Resize({1, 1, 1, 1});
  float *max_in_data = input_tensors[2].mutable_data<float>();
  quantizer_.Quantize(input_tensor, &input_tensors[0], min_in_data, max_in_data);
  if (!ExecuteGraphNew(input_tensors, &output_tensors)) {
    return false;
  }

  output_tensor->ResizeLike(output_tensors[0]);

  const float *min_out_data = output_tensors[1].data<float>();
  const float *max_out_data = output_tensors[2].data<float>();
  quantizer_.DeQuantize(output_tensors[0], *min_out_data, *max_out_data, output_tensor);
  return true;
}

} // namespace mace
