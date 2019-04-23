// Copyright 2019 The MACE Authors. All Rights Reserved.
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

#include "mace/ops/arm/fp32/deconv_2d.h"

#include <utility>
#include <functional>
#include "mace/utils/memory.h"
#include "mace/ops/common/conv_pool_2d_util.h"

namespace mace {
namespace ops {
namespace arm {
namespace fp32 {

MaceStatus Deconv2dBase::ResizeOutAndPadOut(
    const OpContext *context,
    const Tensor *input,
    const Tensor *filter,
    const Tensor *output_shape,
    Tensor *output,
    std::vector<int> *out_pad_size,
    std::unique_ptr<Tensor> *padded_output) {
  std::vector<index_t> out_shape;
  if (output_shape) {
    Tensor::MappingGuard out_shape_guard(output_shape);
    MACE_CHECK(output_shape->size() == 4, "output shape should be 4-dims");
    out_shape =
        std::vector<index_t>(output_shape->data<int32_t>(),
                             output_shape->data<int32_t>() + 4);
  }

  std::vector<index_t> padded_out_shape;

  CalDeconvOutputShapeAndPadSize(input->shape(),
                                 filter->shape(),
                                 strides_,
                                 padding_type_,
                                 paddings_,
                                 group_,
                                 &out_shape,
                                 nullptr,
                                 out_pad_size,
                                 &padded_out_shape,
                                 framework_type_,
                                 DataFormat::NCHW);

  MACE_RETURN_IF_ERROR(output->Resize(out_shape));

  const bool is_out_padded =
      padded_out_shape[2] != out_shape[2]
          || padded_out_shape[3] != out_shape[3];

  if (is_out_padded) {
    index_t padded_out_size =
        std::accumulate(padded_out_shape.begin(),
                        padded_out_shape.end(),
                        1,
                        std::multiplies<index_t>()) * sizeof(float);
    ScratchBuffer *scratch = context->device()->scratch_buffer();
    scratch->Rewind();
    index_t scratch_size = PadAlignSize(padded_out_size);
    scratch->GrowSize(scratch_size);

    std::unique_ptr<Tensor>
        padded_out
        (make_unique<Tensor>(scratch->Scratch(scratch_size), DT_FLOAT));
    padded_out->Reshape(padded_out_shape);
    *padded_output = std::move(padded_out);
  }

  return MaceStatus::MACE_SUCCESS;
}

void Deconv2dBase::UnPadOutput(const Tensor &src,
                               const std::vector<int> &out_pad_size,
                               Tensor *dst) {
  if (dst == &src) return;
  const index_t pad_h = out_pad_size[0] / 2;
  const index_t pad_w = out_pad_size[1] / 2;

  const index_t batch = dst->dim(0);
  const index_t channels = dst->dim(1);
  const index_t height = dst->dim(2);
  const index_t width = dst->dim(3);
  const index_t padded_height = src.dim(2);
  const index_t padded_width = src.dim(3);

  auto padded_out_data = src.data<float>();
  auto out_data = dst->mutable_data<float>();

  for (index_t i = 0; i < batch; ++i) {
    for (index_t j = 0; j < channels; ++j) {
      for (index_t k = 0; k < height; ++k) {
        const float *input_base =
            padded_out_data + ((i * channels + j) * padded_height
                + (k + pad_h)) * padded_width;
        float *output_base =
            out_data + ((i * channels + j) * height + k) * width;
        memcpy(output_base, input_base + pad_w, width * sizeof(float));
      }
    }
  }
}

}  // namespace fp32
}  // namespace arm
}  // namespace ops
}  // namespace mace
