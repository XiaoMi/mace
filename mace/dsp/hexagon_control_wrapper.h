//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_
#define MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_

#include "mace/dsp/hexagon/hexagon_nn.h"
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
  bool SetupGraph(NetDef net_def);
  bool SetupGraph(const std::string &model_file);
  bool ExecuteGraph(const Tensor &input_tensor, Tensor *output_tensor) {
    LOG(INFO) << "Execute graph: " << nn_id_;
    output_tensor->SetDtype(output_data_type_);
    output_tensor->Resize(output_shape_);
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

    MACE_ASSERT(output_shape == output_shape_,
                "wrong output shape inferred");
    MACE_ASSERT(output_bytes == output_tensor->raw_size(),
                "wrong output bytes inferred.");
    return res == 0;
  };

  bool TeardownGraph();
  void PrintLog();
  void PrintGraph();
  void GetPerfInfo();
  void SetDebugLevel(int level);

 private:
  // CAVEAT: Need offset as HVX library reserves some ids
  static constexpr int NODE_ID_OFFSET = 10000;

  uint32_t node_id(uint32_t nodeid) {
    return NODE_ID_OFFSET + nodeid;
  }

  int nn_id_;
  OpMap op_map_;
  Serializer serializer_;

  vector<index_t> input_shape_;
  vector<index_t> output_shape_;
  DataType input_data_type_;
  DataType output_data_type_;

 DISABLE_COPY_AND_ASSIGN(HexagonControlWrapper);
};

}

#endif // MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_
