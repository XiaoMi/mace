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

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/ops/operator.h"
#include "mace/core/registry/ops_registry.h"
#include "mace/ops/common/utils.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/resize_nearest_neighbor.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {
template<typename T>
inline void ResizeImageNCHW(const OpContext *context,
                            const T *images,
                            const index_t batch_size,
                            const index_t in_height,
                            const index_t in_width,
                            const index_t out_height,
                            const index_t out_width,
                            const index_t channels,
                            const float height_scale,
                            const float width_scale,
                            bool align_corners,
                            T *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t c = start1; c < end1; c += step1) {
        const T
            *channel_input_ptr =
            images + (b * channels + c) * in_height * in_width;
        T *channel_output_ptr =
            output + (b * channels + c) * out_height * out_width;
        for (index_t y = 0; y < out_height; ++y) {
          const index_t in_y = std::min(
              (align_corners) ? static_cast<index_t>(roundf(y * height_scale))
                              : static_cast<index_t>(floorf(y * height_scale)),
              in_height - 1);
          for (int x = 0; x < out_width; ++x) {
            const index_t in_x = std::min(
                (align_corners) ? static_cast<index_t>(roundf(x * width_scale))
                                : static_cast<index_t>(floorf(x * width_scale)),
                in_width - 1);
            channel_output_ptr[y * out_width + x] =
                channel_input_ptr[in_y * in_width + in_x];
          }
        }
      }
    }
  }, 0, batch_size, 1, 0, channels, 1);
}

template<DeviceType D, typename T>
class ResizeNearestNeighborOp;

template<typename T>
class ResizeNearestNeighborOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ResizeNearestNeighborOp(OpConstructContext *context)
      : Operation(context),
        align_corners_(Operation::GetOptionalArg<bool>("align_corners", false)),
        height_scale_(Operation::GetOptionalArg<float>("height_scale", 0)),
        width_scale_(Operation::GetOptionalArg<float>("width_scale", 0)) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    MACE_CHECK(input->dim_size() == 4,
               "input must be 4-dimensional.", input->dim_size());

    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t in_height = input->dim(2);
    const index_t in_width = input->dim(3);

    index_t out_height = 0;
    index_t out_width = 0;
    if (height_scale_ > 0) {  // for Caffe and ONNX
      out_height = static_cast<index_t>(height_scale_ * in_height);
      out_width = static_cast<index_t>(width_scale_ * in_width);
    } else {  // for tensor (Tf)
      const Tensor *size = this->Input(1);
      Tensor::MappingGuard size_mapper(size);
      MACE_CHECK(size->dim_size() == 1,
                 "size must be 1-dimensional.", size->dim_size());
      out_height = size->data<int32_t>()[0];
      out_width = size->data<int32_t>()[1];
    }

    MACE_CHECK(out_height > 0 && out_width > 0, out_height, out_width);
    std::vector<index_t> out_shape{batch, channels, out_height, out_width};
    MACE_RETURN_IF_ERROR(output->Resize(out_shape));
    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);
    const T *input_data = input->data<T>();
    T *output_data = output->mutable_data<T>();

    if (out_height == in_height && out_width == in_width) {
      std::copy(input_data,
                input_data + batch * channels * in_height * in_width,
                output_data);
      return MaceStatus::MACE_SUCCESS;
    }

    // Caffe/ONNX's scale is the opposite of ours
    float height_scale = height_scale_ > 0 ? 1 / height_scale_ :
                         common::utils::CalculateResizeScale(in_height,
                                                             out_height,
                                                             align_corners_);
    float width_scale = width_scale_ > 0 ? 1 / width_scale_ :
                        common::utils::CalculateResizeScale(in_width,
                                                            out_width,
                                                            align_corners_);
    ResizeImageNCHW(context,
                    input_data,
                    batch,
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    channels,
                    height_scale,
                    width_scale,
                    align_corners_,
                    output_data);
    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool align_corners_;
  float height_scale_;
  float width_scale_;
};

#ifdef MACE_ENABLE_OPENCL
template<>
class ResizeNearestNeighborOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit ResizeNearestNeighborOp(OpConstructContext *context)
      : Operation(context), dim_(Operation::GetRepeatedArgs<index_t>("dim")),
        height_scale_(Operation::GetOptionalArg<float>("height_scale", 0)),
        width_scale_(Operation::GetOptionalArg<float>("width_scale", 0)) {
    bool align_corners = Operation::GetOptionalArg<bool>(
        "align_corners", false);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ResizeNearestNeighborKernel>(
          align_corners);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4,
               "input must be 4-dimensional.", input->dim_size());

    index_t out_height = 0;
    index_t out_width = 0;
    if (height_scale_ > 0) {  // for Caffe and ONNX
      out_height = static_cast<index_t>(height_scale_ * input->dim(1));
      out_width = static_cast<index_t>(width_scale_ * input->dim(2));
    } else if (dim_.size() < 2) {  // for variable tensor (Tf)
      const Tensor *size = this->Input(1);
      Tensor::MappingGuard size_mapper(size);
      MACE_CHECK(size->dim_size() == 1,
                 "size must be 1-dimensional.", size->dim_size());
      out_height = size->data<int32_t>()[0];
      out_width = size->data<int32_t>()[1];
    } else {  // for const tensor (Tf)
      out_height = dim_[0];
      out_width = dim_[1];
    }

    return kernel_->Compute(context, input, out_height, out_width, output);
  }

 private:
  std::vector<index_t> dim_;
  float height_scale_;
  float width_scale_;
  std::unique_ptr<OpenCLResizeNearestNeighborKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterResizeNearestNeighbor(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "ResizeNearestNeighbor",
                   ResizeNearestNeighborOp, DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "ResizeNearestNeighbor",
                        ResizeNearestNeighborOp, DeviceType::CPU);

  MACE_REGISTER_GPU_OP(op_registry, "ResizeNearestNeighbor",
                       ResizeNearestNeighborOp);
}

}  // namespace ops
}  // namespace mace
