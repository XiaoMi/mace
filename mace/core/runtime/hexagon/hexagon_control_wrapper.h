//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_
#define MACE_DSP_HEXAGON_CONTROL_WRAPPER_H_

#include "mace/core/runtime/hexagon/hexagon_controller.h"
#include "mace/core/runtime/hexagon/quantize.h"
#include "mace/core/common.h"
#include "mace/core/tensor.h"
#include "mace/core/public/mace.h"
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
  bool ExecuteGraph(const Tensor &input_tensor, Tensor *output_tensor);
  bool ExecuteGraphNew(const vector<Tensor>& input_tensors,
                       vector<Tensor> *output_tensors);
  bool ExecuteGraphPreQuantize(const Tensor &input_tensor, Tensor *output_tensor);

  bool TeardownGraph();
  void PrintLog();
  void PrintGraph();
  void GetPerfInfo();
  void ResetPerfInfo();
  void SetDebugLevel(int level);
  void SetGraphMode(int mode);

 private:
  // CAVEAT: Need offset as HVX library reserves some ids
  static constexpr int NODE_ID_OFFSET = 10000;

  inline uint32_t node_id(uint32_t nodeid) {
    return NODE_ID_OFFSET + nodeid;
  }

  int nn_id_;
  Serializer serializer_;
  Quantizer quantizer_;

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
