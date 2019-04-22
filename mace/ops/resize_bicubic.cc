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

#include "mace/ops/resize_bicubic.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "mace/core/operator.h"
#ifdef MACE_ENABLE_OPENCL
#include "mace/ops/opencl/image/resize_bicubic.h"
#endif  // MACE_ENABLE_OPENCL
#include "mace/utils/memory.h"

namespace mace {
namespace ops {

inline const std::shared_ptr<float> InitCoeffsTable() {
  // Allocate and initialize coefficients table using Bicubic
  // convolution algorithm.
  // https://en.wikipedia.org/wiki/Bicubic_interpolation
  auto coeffs_tab = std::shared_ptr<float>(
      new float[(resize_bicubic::kTableSize + 1) * 2],
      std::default_delete<float[]>());
  float *coeffs_tab_ptr = coeffs_tab.get();
  static const float A = -0.75f;
  for (int i = 0; i <= resize_bicubic::kTableSize; ++i) {
    float x = i * 1.0f / resize_bicubic::kTableSize;
    coeffs_tab_ptr[i * 2] = ((A + 2) * x - (A + 3)) * x * x + 1;
    x += 1.0;
    coeffs_tab_ptr[i * 2 + 1] = ((A * x - 5 * A) * x + 8 * A) * x - 4 * A;
  }
  return coeffs_tab;
}

inline const float *GetCoeffsTable() {
  // Static so that we initialize it on first use
  static const std::shared_ptr<float> coeffs_tab = InitCoeffsTable();
  return coeffs_tab.get();
}

inline int64_t Bound(int64_t val, int64_t limit) {
  return std::min<int64_t>(limit - 1ll, std::max<int64_t>(0ll, val));
}

inline void GetWeightsAndIndices(float scale, int64_t out_loc, int64_t limit,
                                 std::vector<float> *weights,
                                 std::vector<int64_t> *indices) {
  auto in_loc = static_cast<int64_t>(scale * out_loc);
  const float delta = scale * out_loc - in_loc;
  const int64_t offset = lrintf(delta * resize_bicubic::kTableSize);
  const float *coeffs_tab = GetCoeffsTable();
  *weights = {coeffs_tab[offset * 2 + 1],
              coeffs_tab[offset * 2],
              coeffs_tab[(resize_bicubic::kTableSize - offset) * 2],
              coeffs_tab[(resize_bicubic::kTableSize - offset) * 2 + 1]};
  *indices = {Bound(in_loc - 1, limit), Bound(in_loc, limit),
              Bound(in_loc + 1, limit), Bound(in_loc + 2, limit)};
}

inline float Interpolate1D(const std::vector<float> &weights,
                           const std::vector<float> &values) {
  return values[0] * weights[0] + values[1] * weights[1] +
      values[2] * weights[2] + values[3] * weights[3];
}

inline void ResizeImage(const OpContext *context,
                        const float *images,
                        const index_t batch_size,
                        const index_t in_height,
                        const index_t in_width,
                        const index_t out_height,
                        const index_t out_width,
                        const index_t channels,
                        const float height_scale,
                        const float width_scale,
                        float *output) {
  utils::ThreadPool
      &thread_pool = context->device()->cpu_runtime()->thread_pool();

  thread_pool.Compute2D([=](index_t start0, index_t end0, index_t step0,
                            index_t start1, index_t end1, index_t step1) {
    for (index_t b = start0; b < end0; b += step0) {
      for (index_t y = start1; y < end1; y += step1) {
        std::vector<float> y_weights;
        std::vector<index_t> y_indices;
        GetWeightsAndIndices(height_scale, y, in_height, &y_weights,
                             &y_indices);
        for (index_t x = 0; x < out_width; ++x) {
          std::vector<float> x_weights;
          std::vector<index_t> x_indices;
          GetWeightsAndIndices(width_scale, x, in_width, &x_weights,
                               &x_indices);

          for (index_t c = 0; c < channels; ++c) {
            // Use a 4x4 patch to compute the interpolated output value at
            // (b, y, x, c).
            const float *channel_input_ptr =
                images + (b * channels + c) * in_height * in_width;
            float *channel_output_ptr =
                output + (b * channels + c) * out_height * out_width;
            std::vector<float> coeff(4, 0.0);
            for (index_t i = 0; i < 4; ++i) {
              const std::vector<float> values = {
                  channel_input_ptr[y_indices[i] * in_width + x_indices[0]],
                  channel_input_ptr[y_indices[i] * in_width + x_indices[1]],
                  channel_input_ptr[y_indices[i] * in_width + x_indices[2]],
                  channel_input_ptr[y_indices[i] * in_width + x_indices[3]]};
              coeff[i] = Interpolate1D(x_weights, values);
            }
            channel_output_ptr[y * out_width + x] =
                Interpolate1D(y_weights, coeff);
          }
        }
      }
    }
  }, 0, batch_size, 1, 0, out_height, 1);
}

template<DeviceType D, class T>
class ResizeBicubicOp;

template<>
class ResizeBicubicOp<DeviceType::CPU, float> : public Operation {
 public:
  explicit ResizeBicubicOp(OpConstructContext *context)
      : Operation(context),
        align_corners_(Operation::GetOptionalArg<bool>("align_corners", false)),
        size_(Operation::GetRepeatedArgs<index_t>("size", {-1, -1})) {}

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

    index_t out_height = size_[0];
    index_t out_width = size_[1];
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
      return MaceStatus::MACE_SUCCESS;
    }

    float height_scale =
        resize_bicubic::CalculateResizeScale(in_height,
                                             out_height,
                                             align_corners_);
    float width_scale =
        resize_bicubic::CalculateResizeScale(in_width,
                                             out_width,
                                             align_corners_);

    ResizeImage(context,
                input_data,
                batch,
                in_height,
                in_width,
                out_height,
                out_width,
                channels,
                height_scale,
                width_scale,
                output_data);

    return MaceStatus::MACE_SUCCESS;
  }

 private:
  bool align_corners_;
  std::vector<index_t> size_;
};

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class ResizeBicubicOp<DeviceType::GPU, T> : public Operation {
 public:
  explicit ResizeBicubicOp(OpConstructContext *context)
      : Operation(context) {
    bool align_corners = Operation::GetOptionalArg<bool>(
        "align_corners", false);
    std::vector<index_t> size = Operation::GetRepeatedArgs<index_t>(
        "size", {-1, -1});
    MACE_CHECK(size.size() == 2);
    if (context->GetOpMemoryType() == MemoryType::GPU_IMAGE) {
      kernel_ = make_unique<opencl::image::ResizeBicubicKernel<T>>(
          align_corners, size[0], size[1]);
    } else {
      MACE_NOT_IMPLEMENTED;
    }
  }
  MaceStatus Run(OpContext *context) override {
    const Tensor *input = this->Input(0);
    Tensor *output = this->Output(0);
    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());

    return kernel_->Compute(context, input, output);
  }

 private:
  std::unique_ptr<OpenCLResizeBicubicKernel> kernel_;
};
#endif  // MACE_ENABLE_OPENCL

void RegisterResizeBicubic(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "ResizeBicubic", ResizeBicubicOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "ResizeBicubic", ResizeBicubicOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "ResizeBicubic", ResizeBicubicOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace
