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

#include "micro/ops/utils/activation.h"

#include "micro/base/logging.h"
#include "micro/base/utils.h"
#include "micro/framework/operator.h"
#include "micro/model/argument.h"

namespace micro {
namespace ops {

Activation::Activation() : type_(TYPE_COUNT) {}

MaceStatus Activation::Init(const framework::Operator *op) {
  const char *atcivation_type = reinterpret_cast<const char *>(
      op->GetRepeatArgByName<uint8_t>("activation"));
  if (atcivation_type == NULL) {
    atcivation_type = "NOOP";
  }
  const float max_limit = op->GetArgByName("max_limit", 0.0f);
  const float leakyrelu_coefficient =
      op->GetArgByName("leakyrelu_coefficient", 0.0f);

  return Init(atcivation_type, max_limit, leakyrelu_coefficient);
}

MaceStatus Activation::Init(const char *type, const float limit,
                            const float leakyrelu_coefficient) {
  type_ = StringToActivationType(type);
  limit_ = limit;
  leakyrelu_coefficient_ = leakyrelu_coefficient;

  return MACE_SUCCESS;
}

ActivationType Activation::GetActivationType() {
  MACE_ASSERT1(type_ != TYPE_COUNT, "Activation should init first.");
  return type_;
}

MaceStatus Activation::Compute(const mifloat *input_ptr,
                               const int32_t size, mifloat *output_ptr) {
  MACE_ASSERT1(type_ != TYPE_COUNT, "Activation should init first.");
  switch (type_) {
    case RELU: {
      for (int32_t i = 0; i < size; ++i) {
        *output_ptr++ = base::max<float>(0.f, *input_ptr++);
      }
      break;
    }
    case RELUX: {
      for (int32_t i = 0; i < size; ++i) {
        *output_ptr++ = base::max(0.f, base::min<float>(limit_, *input_ptr++));
      }
      break;
    }
    case LEAKYRELU: {
      for (int32_t i = 0; i < size; ++i) {
        float input = *input_ptr;
        *output_ptr = base::max(input, 0.f) +
            base::min(input, 0.f) * leakyrelu_coefficient_;  // NOLINT
        ++input_ptr;
        ++output_ptr;
      }
      break;
    }
    case TANH: {
      for (int32_t i = 0; i < size; ++i) {
        *output_ptr++ = base::tanh(*input_ptr++);
      }
      break;
    }
    case SIGMOID: {
      for (int32_t i = 0; i < size; ++i) {
        *output_ptr++ = 1 / (1 + base::exp(-(*input_ptr++)));
      }
      break;
    }
    case NOOP: {
      break;
    }
    default: {
      MACE_NOT_IMPLEMENTED;
    }
  }

  return MACE_SUCCESS;
}

ActivationType Activation::StringToActivationType(const char *type) {
  if (base::strcmp(type, "RELU") == 0) {
    return RELU;
  } else if (base::strcmp(type, "RELUX") == 0) {
    return RELUX;
  } else if (base::strcmp(type, "PRELU") == 0) {
    return PRELU;
  } else if (base::strcmp(type, "TANH") == 0) {
    return TANH;
  } else if (base::strcmp(type, "SIGMOID") == 0) {
    return SIGMOID;
  } else if (base::strcmp(type, "NOOP") == 0) {
    return NOOP;
  } else if (base::strcmp(type, "LEAKYRELU") == 0) {
    return LEAKYRELU;
  } else {
    LOG(FATAL) << "Unknown activation type: " << type;
  }
  return NOOP;
}

}  // namespace ops
}  // namespace micro
