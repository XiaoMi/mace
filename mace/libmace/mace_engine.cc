// Copyright 2018 The MACE Authors. All Rights Reserved.
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

#include "mace/libmace/engines/base_engine.h"
#include "mace/libmace/engines/engine_registry.h"
#include "mace/port/logger.h"
#include "mace/port/port.h"
#include "mace/public/mace.h"
#include "mace/utils/macros.h"
#include "mace/utils/memory.h"
#include "mace/proto/mace.pb.h"

namespace mace {

class MaceEngine::Impl {
 public:
  explicit Impl(const MaceEngineConfig &config)
      : engine_(SmartCreateEngine(config)) {}

  ~Impl() {}

  MaceStatus Init(const MultiNetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data,
                  const int64_t model_data_size,
                  bool *model_data_unused = nullptr);

  MaceStatus Init(const MultiNetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const std::string &model_data_file);

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const unsigned char *model_data,
                  const int64_t model_data_size,
                  bool *model_data_unused = nullptr);

  MaceStatus Init(const NetDef *net_def,
                  const std::vector<std::string> &input_nodes,
                  const std::vector<std::string> &output_nodes,
                  const std::string &model_data_file);

  MaceStatus Run(const std::map<std::string, MaceTensor> &inputs,
                 std::map<std::string, MaceTensor> *outputs,
                 RunMetadata *run_metadata);

  MaceStatus ReleaseIntermediateBuffer();

 private:
  std::unique_ptr<BaseEngine> engine_;

  MACE_DISABLE_COPY_AND_ASSIGN(Impl);
};

MaceStatus MaceEngine::Impl::Init(const MultiNetDef *multi_net_def,
                                  const std::vector<std::string> &input_nodes,
                                  const std::vector<std::string> &output_nodes,
                                  const unsigned char *model_data,
                                  const int64_t model_data_size,
                                  bool *model_data_unused) {
  return engine_->Init(multi_net_def, input_nodes, output_nodes, model_data,
                       model_data_size, model_data_unused);
}

MaceStatus MaceEngine::Impl::Init(const MultiNetDef *multi_net_def,
                                  const std::vector<std::string> &input_nodes,
                                  const std::vector<std::string> &output_nodes,
                                  const std::string &model_data_file) {
  return engine_->Init(multi_net_def, input_nodes,
                       output_nodes, model_data_file);
}

MaceStatus MaceEngine::Impl::Init(
    const NetDef *net_def, const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const unsigned char *model_data,
    const int64_t model_data_size, bool *model_data_unused) {
  return engine_->Init(net_def, input_nodes, output_nodes, model_data,
                       model_data_size, model_data_unused);
}

MaceStatus MaceEngine::Impl::Init(
    const NetDef *net_def,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const std::string &model_data_file) {
  return engine_->Init(net_def, input_nodes, output_nodes, model_data_file);
}

MaceStatus MaceEngine::Impl::Run(
    const std::map<std::string, MaceTensor> &inputs,
    std::map<std::string, MaceTensor> *outputs,
    RunMetadata *run_metadata) {
  return engine_->Forward(inputs, outputs, run_metadata);
}

MaceStatus MaceEngine::Impl::ReleaseIntermediateBuffer() {
  return engine_->ReleaseIntermediateBuffer();
}

MaceEngine::MaceEngine(const MaceEngineConfig &config) :
    impl_(make_unique<MaceEngine::Impl>(config)) {}

MaceEngine::~MaceEngine() = default;

MaceStatus MaceEngine::Init(const MultiNetDef *multi_net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const unsigned char *model_data,
                            const int64_t model_data_size,
                            bool *model_data_unused) {
  return impl_->Init(multi_net_def, input_nodes, output_nodes,
                     model_data, model_data_size, model_data_unused);
}

MaceStatus MaceEngine::Init(const MultiNetDef *multi_net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const std::string &model_data_file) {
  return impl_->Init(multi_net_def, input_nodes, output_nodes, model_data_file);
}

MaceStatus MaceEngine::Init(const NetDef *net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const unsigned char *model_data,
                            int64_t model_data_size,
                            bool *model_data_unused) {
  return impl_->Init(net_def, input_nodes, output_nodes,
                     model_data, model_data_size, model_data_unused);
}

MaceStatus MaceEngine::Init(const NetDef *net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const std::string &model_data_file) {
  return impl_->Init(net_def, input_nodes, output_nodes, model_data_file);
}

MaceStatus MaceEngine::Run(const std::map<std::string, MaceTensor> &inputs,
                           std::map<std::string, MaceTensor> *outputs,
                           RunMetadata *run_metadata) {
  return impl_->Run(inputs, outputs, run_metadata);
}

MaceStatus MaceEngine::Run(const std::map<std::string, MaceTensor> &inputs,
                           std::map<std::string, MaceTensor> *outputs) {
  return impl_->Run(inputs, outputs, nullptr);
}

MaceStatus MaceEngine::Init(const NetDef *net_def,
                            const std::vector<std::string> &input_nodes,
                            const std::vector<std::string> &output_nodes,
                            const unsigned char *model_data,
                            bool *model_data_unused) {
  return impl_->Init(net_def, input_nodes, output_nodes,
                     model_data, -1, model_data_unused);
}

MaceStatus MaceEngine::ReleaseIntermediateBuffer() {
  return impl_->ReleaseIntermediateBuffer();
}


MaceStatus CreateMaceEngineFromProto(
    const unsigned char *model_graph_proto,
    const size_t model_graph_proto_size,
    const unsigned char *model_weights_data,
    const size_t model_weights_data_size,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine) {
  VLOG(1) << "Create MaceEngine from model graph proto and weights data";

  if (engine == nullptr) {
    return MaceStatus::MACE_INVALID_ARGS;
  }

  engine->reset(new mace::MaceEngine(config));

  MaceStatus status = MaceStatus::MACE_RUNTIME_ERROR;
  auto multi_net_def = std::make_shared<MultiNetDef>();
  bool succ = multi_net_def->ParseFromArray(model_graph_proto,
                                            model_graph_proto_size);
  if (succ) {
    VLOG(1) << "It is a multi_net_def.";
    status = (*engine)->Init(
        multi_net_def.get(), input_nodes, output_nodes,
        model_weights_data, model_weights_data_size);
  } else {
    VLOG(1) << "It is a net_def.";
    auto net_def = std::make_shared<NetDef>();
    succ = net_def->ParseFromArray(model_graph_proto, model_graph_proto_size);
    MACE_CHECK(succ, "load NetDef failed.");
    status = (*engine)->Init(
        net_def.get(), input_nodes, output_nodes,
        model_weights_data, model_weights_data_size);
  }

  return status;
}

// Deprecated, will be removed in future version.
MaceStatus CreateMaceEngineFromProto(
    const std::vector<unsigned char> &model_pb,
    const std::string &model_data_file,
    const std::vector<std::string> &input_nodes,
    const std::vector<std::string> &output_nodes,
    const MaceEngineConfig &config,
    std::shared_ptr<MaceEngine> *engine) {
  VLOG(1) << "Create MaceEngine from model pb";
  LOG(WARNING) << "Function deprecated, please change to the new API";
  // load model
  if (engine == nullptr) {
    return MaceStatus::MACE_INVALID_ARGS;
  }

  std::shared_ptr<NetDef> net_def(new NetDef());
  net_def->ParseFromArray(&model_pb[0], model_pb.size());

  engine->reset(new mace::MaceEngine(config));
  MaceStatus status = (*engine)->Init(
      net_def.get(), input_nodes, output_nodes, model_data_file);

  return status;
}

}  // namespace mace
