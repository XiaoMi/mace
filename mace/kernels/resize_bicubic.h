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

#ifndef MACE_KERNELS_RESIZE_BICUBIC_H_
#define MACE_KERNELS_RESIZE_BICUBIC_H_

#include <algorithm>
#include <memory>
#include <vector>

#include "mace/core/future.h"
#include "mace/core/tensor.h"
#include "mace/utils/logging.h"

#ifdef MACE_ENABLE_OPENCL
#include "mace/core/runtime/opencl/cl2_header.h"
#endif  // MACE_ENABLE_OPENCL

namespace mace {
namespace kernels {

static const int64_t kTableSize = (1 << 10);

inline const float* InitCoeffsTable() {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  float* coeffs_tab = new float[(kTableSize + 1) * 2];
  static const double A = -0.75;
  for (int i = 0; i <= kTableSize; ++i) {
    float x = i * 1.0 / kTableSize;
    coeffs_tab[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs_tab[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  }
  return coeffs_tab;
}

inline const float* GetCoeffsTable() {
  // Static so that we initialize it on first use
  static const float* coeffs_tab = InitCoeffsTable();
  return coeffs_tab;
}

inline int64_t Bound(int64_t val, int64_t limit) {
  return std::min<int64_t>(limit - 1ll, std::max<int64_t>(0ll, val));
}

inline void GetWeightsAndIndices(float scale, int64_t out_loc, int64_t limit,
                                 std::array<float, 4>* weights,
                                 std::array<int64_t, 4>* indices) {
  const int64_t in_loc = scale * out_loc;
  const float delta = scale * out_loc - in_loc;
  const int64_t offset = lrintf(delta * kTableSize);
  const float* coeffs_tab = GetCoeffsTable();
  *weights = {{coeffs_tab[offset * 2 + 1], coeffs_tab[offset * 2],
                      coeffs_tab[(kTableSize - offset) * 2],
                      coeffs_tab[(kTableSize - offset) * 2 + 1]}};
  *indices = {{Bound(in_loc - 1, limit), Bound(in_loc, limit),
                      Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)}};
}

inline float Interpolate1D(const std::array<float, 4>& weights,
                           const std::array<float, 4>& values) {
  return values[0] * weights[0] + values[1] * weights[1] +
         values[2] * weights[2] + values[3] * weights[3];
}

inline float CalculateResizeScale(index_t in_size,
                                  index_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}

inline void ResizeImage(const float *images,
             const index_t batch_size,
             const index_t in_height,
             const index_t in_width,
             const index_t out_height,
             const index_t out_width,
             const index_t channels,
             const float height_scale,
             const float width_scale,
             float *output) {
  std::array<float, 4> coeff = {{0.0, 0.0, 0.0, 0.0}};
#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch_size; ++b) {
    for (index_t y = 0; y < out_height; ++y) {
      std::array<float, 4> y_weights;
      std::array<index_t, 4> y_indices;
      GetWeightsAndIndices(height_scale, y, in_height, &y_weights,
                                 &y_indices);
      for (index_t x = 0; x < out_width; ++x) {
        std::array<float, 4> x_weights;
        std::array<index_t, 4> x_indices;
        GetWeightsAndIndices(width_scale, x, in_width, &x_weights,
                             &x_indices);

        for (index_t c = 0; c < channels; ++c) {
          // Use a 4x4 patch to compute the interpolated output value at
          // (b, y, x, c).
          const float *channel_input_ptr =
                  images + (b * channels + c) * in_height * in_width;
          float *channel_output_ptr =
                  output + (b * channels + c) * out_height * out_width;
          for (index_t i = 0; i < 4; ++i) {
            const std::array<float, 4> values = {
              {static_cast<float>(channel_input_ptr
                  [y_indices[i] * in_width + x_indices[0]]),
               static_cast<float>(channel_input_ptr
                  [y_indices[i] * in_width + x_indices[1]]),
               static_cast<float>(channel_input_ptr
                  [y_indices[i] * in_width + x_indices[2]]),
               static_cast<float>(channel_input_ptr
                  [y_indices[i] * in_width + x_indices[3]])}};
            coeff[i] = Interpolate1D(x_weights, values);
          }
          channel_output_ptr[y * out_width + x] =
                  Interpolate1D(y_weights, coeff);
        }
      }
    }
  }
}

struct ResizeBicubicFunctorBase {
  ResizeBicubicFunctorBase(const std::vector<index_t> &size,
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
struct ResizeBicubicFunctor;

template<>
struct ResizeBicubicFunctor<DeviceType::CPU, float>
    : ResizeBicubicFunctorBase {
  ResizeBicubicFunctor(const std::vector<index_t> &size, bool align_corners)
      : ResizeBicubicFunctorBase(size, align_corners) {}

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

    ResizeImage(input_data, batch, in_height, in_width, out_height, out_width,
            channels, height_scale, width_scale, output_data);

    return MACE_SUCCESS;
  }
};

#ifdef MACE_ENABLE_OPENCL
template<typename T>
struct ResizeBicubicFunctor<DeviceType::GPU, T>
    : ResizeBicubicFunctorBase {
  ResizeBicubicFunctor(const std::vector<index_t> &size, bool align_corners)
      : ResizeBicubicFunctorBase(size, align_corners) {}

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

#endif  // MACE_KERNELS_RESIZE_BICUBIC_H_
