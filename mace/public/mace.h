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

const char *MaceVersion();

enum DeviceType { CPU = 0, GPU = 2, HEXAGON = 3, AUTO = 4 };

enum MaceStatus { MACE_SUCCESS = 0, MACE_INVALID_ARGS = 1 };

// MACE input/output tensor
class MaceTensor {
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

class NetDef;
class RunMetadata;

class MaceEngine {
 public:
  explicit MaceEngine(const NetDef *net_def,
                      DeviceType device_type,
                      const std::vector<std::string> &input_nodes,
                      const std::vector<std::string> &output_nodes);
  ~MaceEngine();

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

std::unique_ptr<MaceEngine> CreateMaceEngine(
    const std::string &model_tag,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const char *model_data_file = nullptr,
    const DeviceType device_type = DeviceType::AUTO);

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_H_
