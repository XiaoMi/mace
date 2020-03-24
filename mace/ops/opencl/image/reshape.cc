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
                                  Tensor *output,
                                  const DataFormat op_data_format) {
  MaceStatus succ = i2bkernel_->Compute(context, input,
                                        OpenCLBufferType::IN_OUT_CHANNEL,
                                        0, inter_buffer_.get());
  MACE_RETURN_IF_ERROR(succ);

  if (op_data_format == DataFormat::NCHW) {
		auto in_shape = inter_buffer_->shape();
		Tensor *tmp = new Tensor(context->device()->allocator(), inter_buffer_->dtype());
		Tensor *tmp2 = new Tensor(context->device()->allocator(), inter_buffer_->dtype());

		std::vector<index_t> tmp_shape = {in_shape[0], in_shape[3], in_shape[1], in_shape[2]};
		tmp->Resize(tmp_shape);
		tmp->Reshape(tmp_shape);

		std::vector<index_t> tmp2_shape = {new_shape[0], new_shape[3], new_shape[1], new_shape[2]};
		tmp2->Resize(new_shape);
		tmp2->Reshape(new_shape);

		Tensor::MappingGuard input_guard(inter_buffer_.get());
		Tensor::MappingGuard tmp_guard(tmp);
		Tensor::MappingGuard tmp2_guard(tmp2);

		auto input_data = inter_buffer_.get()->data<float>();
		auto tmp_data = tmp->mutable_data<float>();
		auto tmp2_data = tmp2->mutable_data<float>();

		const float *input_ptr;
		// NHWC->NCHW
		for (index_t i = 0; i < in_shape[3]; i++) {
		  for (index_t j = 0; j < in_shape[1]; j++) {
		    for (index_t k = 0; k < in_shape[2]; k++) {
		      input_ptr = input_data + j*in_shape[2]*in_shape[3] + k*in_shape[3] + i;
		      tmp_data[i*in_shape[2]*in_shape[1] + j*in_shape[2] + k] = *input_ptr;
		    }
		  }
		}

		succ = tmp->Resize(tmp2_shape);
		MACE_RETURN_IF_ERROR(succ);

		const float *tmp_ptr;
		// NCHW->NHWC
		for (index_t i = 0; i < new_shape[3]; i++) {
		  for (index_t j = 0; j < new_shape[1]; j++) {
		    for (index_t k = 0; k < new_shape[2]; k++) {
		      tmp_ptr = tmp_data + i*new_shape[2]*new_shape[1] + j*new_shape[2] + k;
		      tmp2_data[j*new_shape[2]*new_shape[3] + k*new_shape[3] + i] = *tmp_ptr;
		    }
		  }
		}

		succ = b2ikernel_->Compute(context, tmp2,
		                           OpenCLBufferType::IN_OUT_CHANNEL,
		                           0, output);
  }
  else {
    succ = inter_buffer_->Resize(new_shape);
    MACE_RETURN_IF_ERROR(succ);

    succ = b2ikernel_->Compute(context, inter_buffer_.get(),
                               OpenCLBufferType::IN_OUT_CHANNEL,
                               0, output);
  }

  MACE_RETURN_IF_ERROR(succ);

  return MaceStatus::MACE_SUCCESS;
}

}  // namespace image
}  // namespace opencl
}  // namespace ops
}  // namespace mace
