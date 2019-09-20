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

#include "mace/ops/opencl/image/reshape.h"

#include <vector>
#include <memory>

#include "mace/ops/opencl/image/buffer_to_image.h"
#include "mace/ops/opencl/image/image_to_buffer.h"
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
namespace opencl {
namespace image {

ReshapeKernel::ReshapeKernel(OpConstructContext *context) {
  i2bkernel_ = make_unique<opencl::image::ImageToBuffer>();
  b2ikernel_ = make_unique<opencl::image::BufferToImage>();
  inter_buffer_ =
      make_unique<Tensor>(context->device()->allocator(), DT_FLOAT);
  MACE_CHECK(inter_buffer_ != nullptr);
}

MaceStatus ReshapeKernel::Compute(OpContext *context,
                                  const Tensor *input,
                                  const std::vector<index_t> &new_shape,
                                  Tensor *output) {
  MaceStatus succ = i2bkernel_->Compute(context, input,
                                        OpenCLBufferType::IN_OUT_CHANNEL,
                                        0, inter_buffer_.get());
  MACE_RETURN_IF_ERROR(succ);

  succ = inter_buffer_->Resize(new_shape);
  MACE_RETURN_IF_ERROR(succ);

  succ = b2ikernel_->Compute(context, inter_buffer_.get(),
                             OpenCLBufferType::IN_OUT_CHANNEL,
                             0, output);
  MACE_RETURN_IF_ERROR(succ);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
