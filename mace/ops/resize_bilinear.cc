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
#include "mace/utils/memory.h"
#include "mace/core/quantize.h"
#include "mace/ops/common/coordinate_transformation_mode.h"
#include "mace/ops/common/utils.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/resize_bilinear.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace ops {

struct CachedInterpolation {
  index_t lower;  // Lower source index used in the interpolation
  index_t upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

inline void ComputeInterpolationWeights(
    const index_t out_size,
    const index_t in_size,
    const float scale,
    const CoordinateTransformationMode coordinate_transformation_mode,
    CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (index_t i = out_size - 1; i >= 0; --i) {
    float in = i * scale;
    if (coordinate_transformation_mode == HALF_PIXEL) {
      in = (static_cast<float>(i) + 0.5f) * scale - 0.5f;
    } else if (coordinate_transformation_mode == PYTORCH_HALF_PIXEL) {
      in = out_size > 1 ? (static_cast<float>(i) + 0.5f) * scale - 0.5f : 0;
    }
    const float in_f = std::floor(in);
    interpolation[i].lower =
        std::max(static_cast<index_t>(in_f), static_cast<index_t>(0));
    interpolation[i].upper =
        std::min(static_cast<index_t>(std::ceil(in)), in_size - 1);
    interpolation[i].lerp = in - in_f;
  }
}

template<typename T>
inline T ComputeLerp(const T top_left,
                     const T top_right,
                     const T bottom_left,
                     const T bottom_right,
                     const float x_lerp,
                     const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template<>
inline uint8_t ComputeLerp<uint8_t>(const uint8_t top_left,
                                    const uint8_t top_right,
                                    const uint8_t bottom_left,
                                    const uint8_t bottom_right,
                                    const float x_lerp,
                                    const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return Saturate<uint8_t>(roundf(top + (bottom - top) * y_lerp));
}

template<typename T>
inline void ResizeImageNCHW(const OpContext *context,
                            const T *images,
                            const index_t batch_size,
                            const index_t in_height,
                            const index_t in_width,
                            const index_t out_height,
                            const index_t out_width,
                            const index_t channels,
                            const std::vector<CachedInterpolation> &xs_vec,
                            const std::vector<CachedInterpolation> &ys,
                            T *output) {
  const CachedInterpolation *xs = xs_vec.data();

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
          const T *y_lower_input_ptr =
              channel_input_ptr + ys[y].lower * in_width;
          const T *y_upper_input_ptr =
              channel_input_ptr + ys[y].upper * in_width;
          const float ys_lerp = ys[y].lerp;

          for (index_t x = 0; x < out_width; ++x) {
            const float xs_lerp = xs[x].lerp;
            const T top_left = y_lower_input_ptr[xs[x].lower];
            const T top_right = y_lower_input_ptr[xs[x].upper];
            const T bottom_left = y_upper_input_ptr[xs[x].lower];
            const T bottom_right = y_upper_input_ptr[xs[x].upper];
            channel_output_ptr[y * out_width + x] =
                ComputeLerp(top_left, top_right, bottom_left,
                            bottom_right, xs_lerp, ys_lerp);
          }
        }
      }
    }
  }, 0, batch_size, 1, 0, channels, 1);
}

template<typename T>
inline void ResizeImageNHWC(const OpContext *context,
                            const T *images,
                            const index_t batch_size,
                            const index_t in_height,
                            const index_t in_width,
                            const index_t out_height,
                            const index_t out_width,
                            const index_t channels,
                            const std::vector<CachedInterpolation> &xs_vec,
                            const std::vector<CachedInterpolation> &ys,
                            T *output) {
  const CachedInterpolation *xs = xs_vec.data();

  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  for (index_t b = 0; b < batch_size; ++b) {
    const T *input_base = images + b * channels * in_height * in_width;
    T *output_base = output + b * channels * out_height * out_width;

    thread_pool.Compute1D([=](index_t start, index_t end, index_t step) {
      for (index_t y = start; y < end; y += step) {
        const T
            *y_lower_input_ptr = input_base + ys[y].lower * in_width * channels;
        const T
            *y_upper_input_ptr = input_base + ys[y].upper * in_width * channels;
        const float ys_lerp = ys[y].lerp;

        for (index_t x = 0; x < out_width; ++x) {
          const float xs_lerp = xs[x].lerp;
          const T *top_left = y_lower_input_ptr + xs[x].lower * channels;
          const T *top_right = y_lower_input_ptr + xs[x].upper * channels;
          const T *bottom_left = y_upper_input_ptr + xs[x].lower * channels;
          const T *bottom_right = y_upper_input_ptr + xs[x].upper * channels;

          T *output_ptr = output_base + (y * out_width + x) * channels;
          for (index_t c = 0; c < channels; ++c) {
            output_ptr[c] =
                ComputeLerp(top_left[c], top_right[c], bottom_left[c],
                            bottom_right[c], xs_lerp, ys_lerp);
          }
        }
      }
    }, 0, out_height, 1);
  }
}

template<DeviceType D, typename T>
class ResizeBilinearOp;

template<typename T>
class ResizeBilinearOp<DeviceType::CPU, T> : public Operation {
 public:
  explicit ResizeBilinearOp(OpConstructContext *context)
      : Operation(context),
        align_corners_(Operation::GetOptionalArg<bool>("align_corners", false)),
        size_(Operation::GetRepeatedArgs<index_t>("size", {-1, -1})),
        height_scale_(Operation::GetOptionalArg<float>("height_scale", 0)),
        width_scale_(Operation::GetOptionalArg<float>("width_scale", 0)),
        coordinate_transformation_mode_(
            static_cast<CoordinateTransformationMode>(
                Operation::GetOptionalArg<int>("coordinate_transformation_mode",
                                               0))) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(size_.size() == 2);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t in_height = input->dim(2);
    const index_t in_width = input->dim(3);

    index_t out_height = 0;
    index_t out_width = 0;
    if (height_scale_ > 0) {  // for ONNX
      out_height = static_cast<index_t>(height_scale_ * in_height);
      out_width = static_cast<index_t>(width_scale_ * in_width);
    } else {  // for tensor (Tf and Caffe)
      out_height = size_[0];
      out_width = size_[1];
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

    // ONNX's scale is the opposite of ours
    float height_scale = height_scale_ > 0 ? 1 / height_scale_ :
                         common::utils::CalculateResizeScale(in_height,
                                                             out_height,
                                                             align_corners_);
    float width_scale = width_scale_ > 0 ? 1 / width_scale_ :
                        common::utils::CalculateResizeScale(in_width,
                                                            out_width,
                                                            align_corners_);

    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);

    // Compute the cached interpolation weights on the x and y dimensions.
    ComputeInterpolationWeights(out_height, in_height, height_scale,
                                coordinate_transformation_mode_, ys.data());
    ComputeInterpolationWeights(out_width, in_width, width_scale,
                                coordinate_transformation_mode_, xs.data());

    ResizeImageNCHW(context,
                    input_data,
                    batch,
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    channels,
                    xs,
                    ys,
                    output_data);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool align_corners_;
  std::vector<index_t> size_;
  float height_scale_;
  float width_scale_;
  CoordinateTransformationMode coordinate_transformation_mode_;
};

#ifdef MACE_ENABLE_QUANTIZE
template <>
class ResizeBilinearOp<DeviceType::CPU, uint8_t> : public Operation {
 public:
  explicit ResizeBilinearOp(OpConstructContext *context)
      : Operation(context),
        align_corners_(Operation::GetOptionalArg<bool>("align_corners", false)),
        size_(Operation::GetRepeatedArgs<index_t>("size", {-1, -1})),
        height_scale_(Operation::GetOptionalArg<float>("height_scale", 0)),
        width_scale_(Operation::GetOptionalArg<float>("width_scale", 0)),
        coordinate_transformation_mode_(
            static_cast<CoordinateTransformationMode>(
                Operation::GetOptionalArg<int>("coordinate_transformation_mode",
                                               0))) {}

  MaceStatus Run(OpContext *context) override {
    MACE_UNUSED(context);
    MACE_CHECK(size_.size() == 2);
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());
    const index_t batch = input->dim(0);
    const index_t in_height = input->dim(1);
    const index_t in_width = input->dim(2);
    const index_t channels = input->dim(3);

    index_t out_height = 0;
    index_t out_width = 0;
    if (height_scale_ > 0) {  // for ONNX
      out_height = static_cast<index_t>(height_scale_ * in_height);
      out_width = static_cast<index_t>(width_scale_ * in_width);
    } else {  // for tensor (Tf and Caffe)
      out_height = size_[0];
      out_width = size_[1];
    }
    MACE_CHECK(out_height > 0 && out_width > 0);
    std::vector<index_t> out_shape{batch, out_height, out_width, channels};
    MACE_RETURN_IF_ERROR(output->Resize(out_shape));

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);
    const uint8_t *input_data = input->data<uint8_t>();
    uint8_t *output_data = output->mutable_data<uint8_t>();

    if (out_height == in_height && out_width == in_width) {
      std::copy(input_data,
                input_data + batch * in_height * in_width * channels ,
                output_data);
      return MaceStatus::MACE_SUCCESS;
    }

    // ONNX's scale is the opposite of ours
    float height_scale = height_scale_ > 0 ? 1 / height_scale_ :
                         common::utils::CalculateResizeScale(in_height,
                                                             out_height,
                                                             align_corners_);
    float width_scale = width_scale_ > 0 ? 1 / width_scale_ :
                        common::utils::CalculateResizeScale(in_width,
                                                            out_width,
                                                            align_corners_);

    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);

    // Compute the cached interpolation weights on the x and y dimensions.
    ComputeInterpolationWeights(out_height, in_height, height_scale,
                                coordinate_transformation_mode_, ys.data());
    ComputeInterpolationWeights(out_width, in_width, width_scale,
                                coordinate_transformation_mode_, xs.data());

    ResizeImageNHWC(context,
                    input_data,
                    batch,
                    in_height,
                    in_width,
                    out_height,
                    out_width,
                    channels,
                    xs,
                    ys,
                    output_data);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool align_corners_;
  std::vector<index_t> size_;
  float height_scale_;
  float width_scale_;
  CoordinateTransformationMode coordinate_transformation_mode_;
};
#endif  // MACE_ENABLE_QUANTIZE

#ifdef MACE_ENABLE_OPENCL
template<>
class ResizeBilinearOp<DeviceType::GPU, float> : public Operation {
 public:
  explicit ResizeBilinearOp(OpConstructContext *context)
      : Operation(context),
        size_(Operation::GetRepeatedArgs<index_t>("size", {-1, -1})),
        height_scale_(Operation::GetOptionalArg<float>("height_scale", 0)),
        width_scale_(Operation::GetOptionalArg<float>("width_scale", 0))  {
    bool align_corners = Operation::GetOptionalArg<bool>(
        "align_corners", false);
    CoordinateTransformationMode coordinate_transformation_mode =
        static_cast<CoordinateTransformationMode>(
            Operation::GetOptionalArg<int>("coordinate_transformation_mode",
                                           0));
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ResizeBilinearKernel>(
          align_corners, coordinate_transformation_mode);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());

    index_t out_height = 0;
    index_t out_width = 0;
    if (height_scale_ > 0) {  // for ONNX
      out_height = static_cast<index_t>(height_scale_ * input->dim(1));
      out_width = static_cast<index_t>(width_scale_ * input->dim(2));
    } else {  // for tensor (Tf and Caffe)
      out_height = size_[0];
      out_width = size_[1];
    }
    MACE_CHECK(out_height > 0 && out_width > 0);

    return kernel_->Compute(context, input, out_height, out_width, output);
  }

 private:
  std::unique_ptr<OpenCLResizeBilinearKernel> kernel_;
  std::vector<index_t> size_;
  float height_scale_;
  float width_scale_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterResizeBilinear(OpRegistry *op_registry) {
  MACE_REGISTER_OP(op_registry, "ResizeBilinear", ResizeBilinearOp,
                   DeviceType::CPU, float);
  MACE_REGISTER_BF16_OP(op_registry, "ResizeBilinear", ResizeBilinearOp,
                        DeviceType::CPU);

#ifdef MACE_ENABLE_QUANTIZE
  MACE_REGISTER_OP(op_registry, "ResizeBilinear", ResizeBilinearOp,
                   DeviceType::CPU, uint8_t);
#endif  // MACE_ENABLE_QUANTIZE

  MACE_REGISTER_GPU_OP(op_registry, "ResizeBilinear", ResizeBilinearOp);
}

}  // namespace ops
}  // namespace mace
