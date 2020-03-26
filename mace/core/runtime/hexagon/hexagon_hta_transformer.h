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

#ifndef MACE_CORE_RUNTIME_HEXAGON_HEXAGON_HTA_TRANSFORMER_H_
#define MACE_CORE_RUNTIME_HEXAGON_HEXAGON_HTA_TRANSFORMER_H_

#include <memory>
#include <vector>

#include "mace/core/device.h"
#include "mace/core/tensor.h"
#include "mace/core/types.h"
#include "mace/utils/math.h"
#include "mace/utils/thread_pool.h"
#include "third_party/hta/hta_hexagon_api.h"

namespace mace {
class BaseTransformer {
 public:
  BaseTransformer() = default;
  virtual ~BaseTransformer() = default;

  virtual void Init(Device *device) { device_ = device; }
  virtual MaceStatus Compute(const Tensor *input, Tensor *output) = 0;

 protected:
  Device *device_;
};

class HexagonHTATranformerBase {
 public:
  HexagonHTATranformerBase() = default;
  virtual ~HexagonHTATranformerBase() = default;

  virtual void Init(Device *device) = 0;
  virtual MaceStatus SetInputTransformer(
      const hexagon_hta_hw_layout format) = 0;
  virtual MaceStatus SetOutputTransformer(
      const hexagon_hta_hw_layout format) = 0;
  virtual MaceStatus Quantize(const Tensor *input, Tensor *output) = 0;
  virtual MaceStatus Dequantize(const Tensor *input, Tensor *output) = 0;
  virtual MaceStatus TransformInput(const Tensor *input,
                                    Tensor *output,
                                    int index) = 0;
  virtual MaceStatus TransformOutput(const Tensor *input,
                                     Tensor *output,
                                     int index) = 0;
};

template <DeviceType D>
class HexagonHTATranformer : public HexagonHTATranformerBase {
 public:
  void Init(Device *device) override;
  MaceStatus SetInputTransformer(const hexagon_hta_hw_layout format) override;
  MaceStatus SetOutputTransformer(const hexagon_hta_hw_layout format) override;

  MaceStatus Quantize(const Tensor *input, Tensor *output) override;
  MaceStatus Dequantize(const Tensor *input, Tensor *output) override;
  MaceStatus TransformInput(const Tensor *input,
                            Tensor *output,
                            int index) override;
  MaceStatus TransformOutput(const Tensor *input,
                             Tensor *output,
                             int index) override;

 private:
  Device *device_;
  std::unique_ptr<BaseTransformer> quantizer_;
  std::unique_ptr<BaseTransformer> dequantizer_;
  std::vector<std::unique_ptr<BaseTransformer>> input_transformers_;
  std::vector<std::unique_ptr<BaseTransformer>> output_transformers_;
};
}  // namespace mace
#endif  // MACE_CORE_RUNTIME_HEXAGON_HEXAGON_HTA_TRANSFORMER_H_
