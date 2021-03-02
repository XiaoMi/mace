// Copyright 2020 The MACE Authors. All Rights Reserved.
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

#ifndef MACE_CORE_FLOW_BASE_FLOW_H_
#define MACE_CORE_FLOW_BASE_FLOW_H_


#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "mace/core/flow/flow_registry.h"
#include "mace/core/net/base_net.h"
#include "mace/core/workspace.h"
#include "mace/public/mace.h"
#include "mace/utils/macros.h"

namespace mace {

namespace utils {
class ThreadPool;
}  // namespace utils
class BaseEngine;
class MaceEngineCfgImpl;
class Runtime;
class NetDef;
class Tensor;
class OpRegistry;

struct FlowContext {
  MaceEngineCfgImpl *config_impl;
  OpRegistry *op_registry;
  OpDelegatorRegistry *op_delegator_registry;
  Runtime *cpu_runtime;
  Runtime *main_runtime;
  utils::ThreadPool *thread_pool;
  BaseEngine *parent_engine;

  FlowContext(MaceEngineCfgImpl *cfg_impl, OpRegistry *op_reg,
              OpDelegatorRegistry *op_delegator_reg, Runtime *cpu_rt,
              Runtime *main_rt, utils::ThreadPool *thrd_pool,
              BaseEngine *engine)
      : config_impl(cfg_impl), op_registry(op_reg),
        op_delegator_registry(op_delegator_reg), cpu_runtime(cpu_rt),
        main_runtime(main_rt), thread_pool(thrd_pool), parent_engine(engine) {}
};

class BaseFlow {
 public:
  typedef std::map<std::string, Tensor *> TensorMap;
  explicit BaseFlow(FlowContext *flow_context);
  virtual ~BaseFlow() = default;

  const std::string &GetName() const;
  const BaseEngine *GetMaceEngine() const;

  virtual MaceStatus Init(const NetDef *net_def,
                          const unsigned char *model_data,
                          const int64_t model_data_size,
                          bool *model_data_unused);

  virtual MaceStatus Run(TensorMap *input_tensors,
                         TensorMap *output_tensors,
                         RunMetadata *run_metadata) = 0;

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata = nullptr);

  MaceStatus AllocateIntermediateBuffer();

 protected:
  virtual MaceStatus GetInputTransposeDims(
      const std::pair<const std::string, MaceTensor> &input,
      const Tensor *input_tensor, std::vector<int> *dst_dims,
      DataFormat *data_format);
  virtual MaceStatus TransposeInputByDims(const MaceTensor &mace_tensor,
                                          Tensor *input_tensor,
                                          const std::vector<int> &dst_dims);
  virtual MaceStatus TransposeOutputByDims(const mace::Tensor &output_tensor,
                                           MaceTensor *mace_tensor,
                                           const std::vector<int> &dst_dims);

  MaceStatus InitOutputTensor();

 private:
  MaceStatus InitInputTensors();
  MaceStatus AllocateBufferForInputTensors();
  MaceStatus TransposeInput(
      const std::pair<const std::string, MaceTensor> &input,
      Tensor *input_tensor);

  std::vector<int> GetOutputTransposeDims(
      const mace::Tensor &output_tensor,
      std::pair<const std::string, mace::MaceTensor> *output);
  MaceStatus TransposeOutput(
      const mace::Tensor &output_tensor,
      std::pair<const std::string, mace::MaceTensor> *output);

  Tensor *CreateInputTensor(const std::string &input_name,
                            DataType input_dt);

  MACE_DISABLE_COPY_AND_ASSIGN(BaseFlow);

 protected:
  std::string name_;
  std::unique_ptr<BaseNet> net_;
  std::unique_ptr<Workspace> ws_;
  bool is_quantized_model_;
  DataType net_data_type_;
  std::unordered_map<std::string, mace::InputOutputInfo> input_info_map_;
  std::unordered_map<std::string, mace::InputOutputInfo> output_info_map_;

  // objects not retain
  OpRegistry *op_registry_;
  MaceEngineCfgImpl *config_impl_;
  Runtime *cpu_runtime_;
  Runtime *main_runtime_;
  utils::ThreadPool *thread_pool_;
  BaseEngine *parent_engine_;
};

}  // namespace mace

#endif  // MACE_CORE_FLOW_BASE_FLOW_H_
