//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

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

enum DeviceType { CPU = 0, NEON = 1, OPENCL = 2, HEXAGON = 3 };

enum MaceStatus {
  MACE_SUCCESS = 0,
  MACE_INVALID_ARGS = 1,
  MACE_OUT_OF_RESOURCES = 2,
};

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
  explicit MaceEngine(DeviceType device_type);

  ~MaceEngine();

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes);

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

}  // namespace mace

#endif  // MACE_PUBLIC_MACE_H_
