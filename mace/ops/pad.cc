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

#include <vector>
#include <memory>

#include "mace/core/operator.h"
#include "mace/ops/pad.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/pad.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"
#include "mace/utils/math.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class PadOp;

template <typename T>
class PadOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit PadOp(OpConstructContext *context)
      : Operation(context),
        type_(
            static_cast<PadType>(Operation::GetOptionalArg<int>(
                "pad_type", static_cast<int>(PadType::CONSTANT)))),
        paddings_(Operation::GetRepeatedArgs<int>("paddings")),
        constant_value_(Operation::GetOptionalArg<float>(
            "constant_value", 0.0)) {
    MACE_CHECK(paddings_.size() == 8);
    auto has_df = Operation::GetOptionalArg<int>(
        "has_data_format", 0);
    if (has_df) {
      paddings_ = TransposeShape<int, int>(paddings_, {0, 1, 6, 7, 2, 3, 4, 5});
    }
  }

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(
        this->paddings_.size() == static_cast<size_t>(input->dim_size()) * 2);
    auto input_shape = input->shape();
    for (size_t i = 0; i < paddings_.size(); ++i) {
      if (type_ == PadType::REFLECT || type_ == PadType::SYMMETRIC) {
        MACE_CHECK(paddings_[i] < input_shape[i / 2], paddings_[i],
                   " vs ", input_shape[i / 2]);
      }
      MACE_CHECK(paddings_[i] >= 0);
    }
    MACE_RETURN_IF_ERROR(output->Resize({input_shape[0] + this->paddings_[0]
                                             + this->paddings_[1],
                                         input_shape[1] + this->paddings_[2]
                                             + this->paddings_[3],
                                         input_shape[2] + this->paddings_[4]
                                             + this->paddings_[5],
                                         input_shape[3] + this->paddings_[6]
                                             + this->paddings_[7]}));

    Tensor::MappingGuard input_guard(input);
    Tensor::MappingGuard output_guard(output);
    auto input_ptr = input->data<T>();
    T *output_ptr = output->mutable_data<T>();

    const index_t batch = input->dim(0);
    const index_t channel = input->dim(1);
    const index_t height = input->dim(2);
    const index_t width = input->dim(3);

    if (type_ == PadType::CONSTANT) {
      std::fill(output_ptr, output_ptr + output->size(), this->constant_value_);

      for (index_t b = 0; b < batch; ++b) {
        for (index_t c = 0; c < channel; ++c) {
          for (index_t h = 0; h < height; ++h) {
            const index_t in_offset = (((b * channel + c) * height) +
                                      h) * width;
            const index_t out_offset =
                  (((b + this->paddings_[0]) * output->dim(1)
                + (c + this->paddings_[2])) * output->dim(2)
                + (h + this->paddings_[4])) * output->dim(3)
                + this->paddings_[6];
            memcpy(output_ptr + out_offset,
                   input_ptr + in_offset,
                   width * sizeof(T));
          }
        }
      }
    } else if (type_ == PadType::REFLECT || type_ == PadType::SYMMETRIC) {
      const index_t o_batch   = output->dim(0);
      const index_t o_channel = output->dim(1);
      const index_t o_height  = output->dim(2);
      const index_t o_width   = output->dim(3);
      const int l_add = type_ == PadType::REFLECT ?  0 : -1;
      const int r_add = type_ == PadType::REFLECT ? -2 : -1;

      for (index_t h = 0; h < o_height; ++h) {
        index_t h_in = get_src_idx(h, height, paddings_[4], l_add, r_add);

        for (index_t b = 0; b < o_batch; ++b) {
          index_t b_in = get_src_idx(b, batch, paddings_[0], l_add, r_add);

          for (index_t c = 0; c < o_channel; ++c) {
            index_t c_in = get_src_idx(c, channel, paddings_[2], l_add, r_add);
            const index_t in_offset = (((b_in * channel + c_in) * height) +
                                      h_in) * width;
            index_t out_offset = (((b * o_channel + c) * o_height) +
                                 h) * o_width;

            for (index_t i = 0, j = paddings_[6] + l_add;
                 i < paddings_[6]; ++i, --j) {
              output_ptr[out_offset++] = input_ptr[in_offset + j];
            }
            memcpy(output_ptr + out_offset, input_ptr + in_offset,
                   width * sizeof(T));
            out_offset += width;
            for (index_t i = 0, j = width + r_add; i < paddings_[7]; ++i, --j) {
              output_ptr[out_offset++] = input_ptr[in_offset + j];
            }
          }
        }
      }
    } else {
      LOG(FATAL) << "Pad op doesn't support type " << type_;
    }

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  int get_src_idx(int out, int in_size, int pad, int l_add, int r_add) {
    const int diff_left = pad - out;
    int in;

    if (diff_left > 0) {
      in = diff_left + l_add;

    } else {
      const int diff_right = out - (in_size + pad);

      if (diff_right >= 0) {
        in = in_size - diff_right + r_add;

      } else {
        in = -diff_left;
      }
    }

    return in;
  }

  PadType type_;
  std::vector<int> paddings_;
  float constant_value_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class PadOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit PadOp(OpConstructContext *context)
      : Operation(context) {
    PadType type = static_cast<PadType>(Operation::GetOptionalArg<int>(
      "pad_type", static_cast<int>(PadType::CONSTANT)));
    std::vector<int> paddings = Operation::GetRepeatedArgs<int>("paddings");
    float constant_value = Operation::GetOptionalArg<float>(
        "constant_value", 0.0);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::PadKernel<T>>(
          type, paddings, constant_value);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLPadKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL


void RegisterPad(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "Pad", PadOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "Pad", PadOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "Pad", PadOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
