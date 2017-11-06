//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include "mace/dsp/hexagon_control_wrapper.h"
#include <fstream>

namespace mace {

#define MAX_NODE 2048 * 8

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
  return hexagon_nn_config();
}

bool HexagonControlWrapper::Init() {
  LOG(INFO) << "Hexagon init";
  hexagon_controller_InitHexagonWithMaxAttributes(0, 100);
  nn_id_ = hexagon_nn_init();
  ResetPerfInfo();
  return true;
}

bool HexagonControlWrapper::Finalize() {
  LOG(INFO) << "Hexagon finalize";
  hexagon_controller_DeInitHexagon();
  return true;
}

bool HexagonControlWrapper::SetupGraph(NetDef net_def) {
  LOG(INFO) << "Hexagon setup graph";
  // const node
  for (const TensorProto& tensor_proto: net_def.tensors()) {
    vector<int> tensor_shape(tensor_proto.dims().begin(),
                             tensor_proto.dims().end());
    while (tensor_shape.size() < 4) {
      tensor_shape.insert(tensor_shape.begin(), 1);
    }

    if (tensor_proto.data_type() == DataType::DT_INT32
        && tensor_proto.int32_data_size() == 0) {
      hexagon_nn_append_const_node(nn_id_, node_id(tensor_proto.node_id()),
                                   tensor_shape[0], tensor_shape[1],
                                   tensor_shape[2], tensor_shape[3],
                                   NULL,
                                   0);
    } else {
      unique_ptr<Tensor> tensor = serializer_.Deserialize(tensor_proto,
                                                          DeviceType::CPU);
      VLOG(0) << "Tensor size: " << tensor->size();
      hexagon_nn_append_const_node(nn_id_, node_id(tensor_proto.node_id()),
                                   tensor_shape[0], tensor_shape[1],
                                   tensor_shape[2], tensor_shape[3],
                                   reinterpret_cast<const unsigned char *>(
                                       tensor->raw_data()),
                                   tensor->raw_size());
    }
    VLOG(0) << "Const: " << tensor_proto.name()
            << ", node_id: " << node_id(tensor_proto.node_id())
            << "\n\t shape: " << tensor_shape[0] << " " << tensor_shape[1]
            << " " << tensor_shape[2] << " " << tensor_shape[3];
  }

  // op node
  for (const OperatorDef& op: net_def.op()) {
    unsigned int op_id;
    MACE_CHECK(hexagon_nn_op_name_to_id(op.type().data(), &op_id) == 0,
               "invalid op: ", op.name(), ", type: ", op.type());
    vector<hexagon_nn_input> inputs(op.node_input_size());
    for (size_t i = 0; i < op.node_input_size(); ++i) {
      inputs[i].src_id = node_id(op.node_input(i).node_id());
      inputs[i].output_idx = op.node_input(i).output_port();
    }
    vector<hexagon_nn_output> outputs(op.out_max_byte_size_size());
    for (size_t i = 0; i < op.out_max_byte_size_size(); ++i) {
      outputs[i].max_size = op.out_max_byte_size(i);
    }

    hexagon_nn_padding_type padding_type = static_cast<hexagon_nn_padding_type>(
        op.padding());

    hexagon_nn_append_node(nn_id_, node_id(op.node_id()), op_id, padding_type,
                           inputs.data(), inputs.size(),
                           outputs.data(), outputs.size());

    VLOG(0) << "Op: " << op.name()
            << ", type: " << op.type()
            << ", node_id: " << node_id(op.node_id())
            << ", padding_type: " << padding_type;
    for (const auto& input: inputs) {
      VLOG(0) << "\t input: " << input.src_id << ":" << input.output_idx;
    }
    for (const auto& output: outputs) {
      VLOG(0) << "\t output: " << output.max_size;
    }
  }

  // input info
  const InputInfo& input_info = net_def.input_info()[0];
  input_shape_.insert(input_shape_.begin(),
                      input_info.dims().begin(), input_info.dims().end());
  while (input_shape_.size() < 4) {
    input_shape_.insert(input_shape_.begin(), 1);
  }
  input_data_type_ = input_info.data_type();

  // output info
  const OutputInfo& output_info = net_def.output_info()[0];
  output_shape_.insert(output_shape_.begin(),
                       output_info.dims().begin(), output_info.dims().end());
  while (output_shape_.size() < 4) {
    output_shape_.insert(output_shape_.begin(), 1);
  }
  output_data_type_ = output_info.data_type();

  bool res =  hexagon_nn_prepare(nn_id_) == 0;
  return res;
}

bool HexagonControlWrapper::SetupGraph(const std::string& model_file) {
  std::ifstream file_stream(model_file, std::ios::in | std::ios::binary);
  NetDef net_def;
  net_def.ParseFromIstream(&file_stream);
  file_stream.close();
  return SetupGraph(net_def);
}

bool HexagonControlWrapper::TeardownGraph() {
  LOG(INFO) << "Hexagon teardown graph";
  return hexagon_nn_teardown(nn_id_) == 0;
}

#define PRINT_BUFSIZE (2*1024*1024)

void HexagonControlWrapper::PrintLog() {
  LOG(INFO) << "Print Log";
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

void HexagonControlWrapper::GetPerfInfo() {
  LOG(INFO) << "Get perf info";
  vector<hexagon_nn_perfinfo> perf_info(MAX_NODE);
  unsigned int n_items;
  hexagon_nn_get_perfinfo(nn_id_, perf_info.data(), MAX_NODE, &n_items);

  std::unordered_map<uint32_t, float> node_id_counters;
  std::unordered_map<std::string, std::pair<int, float>> node_type_counters;
  float total_duration = 0.0;
  for (int i = 0; i < n_items; ++i) {
    unsigned int node_id = perf_info[i].node_id;
    unsigned int node_type_id = perf_info[i].node_type;
    node_id_counters[node_id] = ((static_cast<uint64_t>(perf_info[i].counter_hi) << 32)
        + perf_info[i].counter_lo) * 1.0f / perf_info[i].executions;

    LOG(INFO) << "node id: " << perf_info[i].node_id
              << ", node type: " << perf_info[i].node_type
              << ", executions: " << perf_info[i].executions
              << ", duration: " << node_id_counters[node_id];


    char node_type_buf[1280];
    hexagon_nn_op_id_to_name(node_type_id, node_type_buf, 1280);
    std::string node_type(node_type_buf);
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

} // namespace mace