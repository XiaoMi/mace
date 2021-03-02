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

#ifndef MACE_FLOWS_APU_APU_REF_FLOW_H_
#define MACE_FLOWS_APU_APU_REF_FLOW_H_

#include <string>
#include <utility>
#include <vector>

#include "mace/core/flow/common_fp32_flow.h"

namespace mace {

class ApuRefFlow : public CommonFp32Flow {
 public:
  explicit ApuRefFlow(FlowContext *flow_context);
  virtual ~ApuRefFlow() = default;

  MaceStatus Init(const NetDef *net_def,
                  const unsigned char *model_data,
                  const int64_t model_data_size,
                  bool *model_data_unused) override;

  MaceStatus Run(TensorMap *input_tensors,
                 TensorMap *output_tensors,
                 RunMetadata *run_metadata) override;

 protected:
  MaceStatus GetInputTransposeDims(
      const std::pair<const std::string, MaceTensor> &input,
      const Tensor *input_tensor, std::vector<int> *dst_dims,
      DataFormat *data_format) override;

 private:
  MACE_DISABLE_COPY_AND_ASSIGN(ApuRefFlow);
};

}  // namespace mace

#endif  // MACE_FLOWS_APU_APU_REF_FLOW_H_
