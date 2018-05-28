// Copyright 2018 Xiaomi, Inc.  All rights reserved.
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

#ifndef MACE_KERNELS_RESIZE_BILINEAR_H_
#define MACE_KERNELS_RESIZE_BILINEAR_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

struct CachedInterpolation {
  index_t lower;  // Lower source index used in the interpolation
  index_t upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

inline float CalculateResizeScale(index_t in_size,
                                  index_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}

inline void ComputeInterpolationWeights(
    const index_t out_size,
    const index_t in_size,
    const float scale,
    CachedInterpolation *interpolation) {
  interpolation[out_size].lower = 0;
  interpolation[out_size].upper = 0;
  for (index_t i = out_size - 1; i >= 0; --i) {
    const float in = i * scale;
    interpolation[i].lower = static_cast<index_t>(in);
    interpolation[i].upper = std::min(interpolation[i].lower + 1, in_size - 1);
    interpolation[i].lerp = in - interpolation[i].lower;
  }
}

inline float ComputeLerp(const float top_left,
                         const float top_right,
                         const float bottom_left,
                         const float bottom_right,
                         const float x_lerp,
                         const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

inline void ResizeImage(const float *images,
                        const index_t batch_size,
                        const index_t in_height,
                        const index_t in_width,
                        const index_t out_height,
                        const index_t out_width,
                        const index_t channels,
                        const std::vector<CachedInterpolation> &xs_vec,
                        const std::vector<CachedInterpolation> &ys,
                        float *output) {
  const CachedInterpolation *xs = xs_vec.data();

#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch_size; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const float
          *channel_input_ptr =
          images + (b * channels + c) * in_height * in_width;
      float *channel_output_ptr =
          output + (b * channels + c) * out_height * out_width;
      for (index_t y = 0; y < out_height; ++y) {
        const float *y_lower_input_ptr =
            channel_input_ptr + ys[y].lower * in_width;
        const float *y_upper_input_ptr =
            channel_input_ptr + ys[y].upper * in_width;
        const float ys_lerp = ys[y].lerp;

        for (index_t x = 0; x < out_width; ++x) {
          const float xs_lerp = xs[x].lerp;
          const float top_left = y_lower_input_ptr[xs[x].lower];
          const float top_right = y_lower_input_ptr[xs[x].upper];
          const float bottom_left = y_upper_input_ptr[xs[x].lower];
          const float bottom_right = y_upper_input_ptr[xs[x].upper];
          channel_output_ptr[y * out_width + x] =
              ComputeLerp(top_left, top_right, bottom_left,
                          bottom_right, xs_lerp, ys_lerp);
        }
      }
    }
  }
}

struct ResizeBilinearFunctorBase {
  ResizeBilinearFunctorBase(const std::vector<index_t> &size,
                            bool align_corners)
      : align_corners_(align_corners) {
    MACE_CHECK(size.size() == 2);
    out_height_ = size[0];
    out_width_ = size[1];
  }

 protected:
  bool align_corners_;
  index_t out_height_;
  index_t out_width_;
};

template<DeviceType D, typename T>
struct ResizeBilinearFunctor;

template<>
struct ResizeBilinearFunctor<DeviceType::CPU, float>
    : ResizeBilinearFunctorBase {
  ResizeBilinearFunctor(const std::vector<index_t> &size, bool align_corners)
      : ResizeBilinearFunctorBase(size, align_corners) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future) {
    MACE_UNUSED(future);
    const index_t batch = input->dim(0);
    const index_t channels = input->dim(1);
    const index_t in_height = input->dim(2);
    const index_t in_width = input->dim(3);

    index_t out_height = out_height_;
    index_t out_width = out_width_;
    MACE_CHECK(out_height > 0 && out_width > 0);
    std::vector<index_t> out_shape{batch, channels, out_height, out_width};
    MACE_RETURN_IF_ERROR(output->Resize(out_shape));

    Tensor::MappingGuard input_mapper(input);
    Tensor::MappingGuard output_mapper(output);
    const float *input_data = input->data<float>();
    float *output_data = output->mutable_data<float>();

    if (out_height == in_height && out_width == in_width) {
      std::copy(input_data,
                input_data + batch * channels * in_height * in_width,
                output_data);
      return MACE_SUCCESS;
    }

    float height_scale =
        CalculateResizeScale(in_height, out_height, align_corners_);
    float width_scale =
        CalculateResizeScale(in_width, out_width, align_corners_);

    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);

    // Compute the cached interpolation weights on the x and y dimensions.
    ComputeInterpolationWeights(out_height, in_height, height_scale, ys.data());
    ComputeInterpolationWeights(out_width, in_width, width_scale, xs.data());

    ResizeImage(input_data, batch, in_height, in_width, out_height, out_width,
                channels, xs, ys, output_data);

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct ResizeBilinearFunctor<DeviceType::GPU, T>
    : ResizeBilinearFunctorBase {
  ResizeBilinearFunctor(const std::vector<index_t> &size, bool align_corners)
      : ResizeBilinearFunctorBase(size, align_corners) {}

  MaceStatus operator()(const Tensor *input,
                        Tensor *output,
                        StatsFuture *future);

  cl::Kernel kernel_;
  uint32_t kwg_size_;
  std::unique_ptr<BufferBase> kernel_error_;
  std::vector<index_t> input_shape_;
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace kernels
}  // namespace mace

#endif  // MACE_KERNELS_RESIZE_BILINEAR_H_
