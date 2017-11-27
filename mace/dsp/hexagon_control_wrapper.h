//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_
#define MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_

#include "mace/dsp/hexagon/hexagon_controller.h"
#include "mace/dsp/hexagon_nn_ops.h"
#include "mace/core/common.h"
#include "mace/core/tensor.h"
#include "mace/proto/mace.pb.h"
#include "mace/core/serializer.h"

namespace mace {

class HexagonControlWrapper {
 public:
  HexagonControlWrapper() {};
  int GetVersion();
  bool Config();
  bool Init();
  bool Finalize();
  bool SetupGraph(const NetDef& net_def);
  bool SetupGraph(const std::string &model_file);
  bool ExecuteGraph(const Tensor &input_tensor, Tensor *output_tensor) {
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

  bool ExecuteGraphNew(const Tensor *input_tensors, int num_inputs,
                       Tensor *output_tensors, int num_outputs) {
    LOG(INFO) << "Execute graph new: " << nn_id_;
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
      output_tensors[i].SetDtype(output_data_types_[i]);
      output_tensors[i].Resize(output_shapes_[i]);
      vector<index_t> output_shape = output_tensors[0].shape();
      outputs[i].batches = output_shape[0];
      outputs[i].height = output_shape[1];
      outputs[i].width = output_shape[2];
      outputs[i].depth = output_shape[3];
      outputs[i].data = reinterpret_cast<unsigned char *>(
          output_tensors[i].raw_mutable_data());
      outputs[i].dataLen = output_tensors[i].raw_size();
      outputs[i].data_valid_len = output_tensors[i].raw_size();
      outputs[i].unused = 0;
    }

    int res = hexagon_nn_execute_new(nn_id_, inputs, num_inputs,
                                     outputs, num_outputs);

    delete(inputs);
    delete(outputs);
    return res == 0;
  };

  bool TeardownGraph();
  void PrintLog();
  void PrintGraph();
  void GetPerfInfo();
  void ResetPerfInfo();
  void SetDebugLevel(int level);

 private:
  // CAVEAT: Need offset as HVX library reserves some ids
  static constexpr int NODE_ID_OFFSET = 10000;

  inline uint32_t node_id(uint32_t nodeid) {
    return NODE_ID_OFFSET + nodeid;
  }

  int nn_id_;
  Serializer serializer_;

  vector<vector<index_t>> input_shapes_;
  vector<vector<index_t>> output_shapes_;
  vector<DataType> input_data_types_;
  vector<DataType> output_data_types_;
  uint32_t num_inputs_;
  uint32_t num_outputs_;

 DISABLE_COPY_AND_ASSIGN(HexagonControlWrapper);
};

}

#endif // MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_
