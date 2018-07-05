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

// This file defines core MACE APIs.
// There APIs will be stable and backward compatible.

#ifndef MACE_PUBLIC_MACE_H_
#define MACE_PUBLIC_MACE_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mace {

class NetDef;

enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3 };

struct CallStats {
  int64_t start_micros;
  int64_t end_micros;
};

struct ConvPoolArgs {
  std::vector<int> strides;
  int padding_type;
  std::vector<int> paddings;
  std::vector<int> dilations;
  std::vector<int64_t> kernels;
};

struct OperatorStats {
  std::string operator_name;
  std::string type;
  std::vector<std::vector<int64_t>> output_shape;
  ConvPoolArgs args;
  CallStats stats;
};

class RunMetadata {
 public:
  std::vector<OperatorStats> op_stats;
};

const char *MaceVersion();

enum MaceStatus {
  MACE_SUCCESS = 0,
  MACE_INVALID_ARGS = 1,
  MACE_OUT_OF_RESOURCES = 2
};

#define MACE_RETURN_IF_ERROR(stmt)                                          \
  {                                                                        \
    MaceStatus status = (stmt);                                            \
    if (status != MACE_SUCCESS) {                                          \
      VLOG(0) << "Mace runtime failure: " << __FILE__ << ":" << __LINE__;  \
      return status;                                                       \
    }                                                                      \
  }

// MACE input/output tensor
class __attribute__((visibility("default"))) MaceTensor {
 public:
  // shape - the shape of the tensor, with size n
  // data - the buffer of the tensor, must not be null with size equals
  //        shape[0] * shape[1] * ... * shape[n-1]
  explicit MaceTensor(const std::vector<int64_t> &shape,
                      std::shared_ptr<float> data);
  MaceTensor();
  MaceTensor(const MaceTensor &other);
  MaceTensor(const MaceTensor &&other);
  MaceTensor &operator=(const MaceTensor &other);
  MaceTensor &operator=(const MaceTensor &&other);
  ~MaceTensor();

  const std::vector<int64_t> &shape() const;
  const std::shared_ptr<float> data() const;
  std::shared_ptr<float> data();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class __attribute__((visibility("default"))) MaceEngine {
 public:
  explicit MaceEngine(DeviceType device_type);
  ~MaceEngine();

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data);

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const std::string &model_data_file);

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs);

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;

  MaceEngine(const MaceEngine &) = delete;
  MaceEngine &operator=(const MaceEngine &) = delete;
};

__attribute__((visibility("default")))
MaceStatus CreateMaceEngineFromProto(
    const std::vector<unsigned char> &model_pb,
    const std::string &model_data_file,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const DeviceType device_type,
    std::shared_ptr<MaceEngine> *engine);

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_H_
