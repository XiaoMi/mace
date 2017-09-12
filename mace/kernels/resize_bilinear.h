//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_KERNELS_RESIZE_BILINEAR_H_
#define MACE_KERNELS_RESIZE_BILINEAR_H_

#include "mace/core/tensor.h"

namespace mace {
namespace kernels {

namespace {
struct CachedInterpolation {
  index_t lower;  // Lower source index used in the interpolation
  index_t upper;  // Upper source index used in the interpolation
  // 1-D linear iterpolation scale (see:
  // https://en.wikipedia.org/wiki/Bilinear_interpolation)
  float lerp;
};

inline float CalculateResizeScale(index_t in_size, index_t out_size,
                                  bool align_corners) {
  return (align_corners && out_size > 1)
         ? (in_size - 1) / static_cast<float>(out_size - 1)
         : in_size / static_cast<float>(out_size);
}

inline void ComputeInterpolationWeights(const index_t out_size,
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

inline float ComputeLerp(const float top_left, const float top_right,
                          const float bottom_left, const float bottom_right,
                          const float x_lerp, const float y_lerp) {
  const float top = top_left + (top_right - top_left) * x_lerp;
  const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
  return top + (bottom - top) * y_lerp;
}

template<typename T>
void ResizeImage(const T *images,
                 const index_t batch_size, const index_t in_height,
                 const index_t in_width, const index_t out_height,
                 const index_t out_width, const index_t channels,
                 const std::vector<CachedInterpolation> &xs_vec,
                 const std::vector<CachedInterpolation> &ys,
                 float *output) {
  const index_t in_channel_size = in_height * in_width;
  const index_t in_batch_num_values = channels * in_channel_size;
  const index_t out_channel_size = out_height * out_width;
  const index_t out_batch_num_values = channels * out_channel_size;
  const CachedInterpolation *xs = xs_vec.data();

#pragma omp parallel for collapse(2)
  for (index_t b = 0; b < batch_size; ++b) {
    for (index_t c = 0; c < channels; ++c) {
      const T* input_ptr = images + in_batch_num_values * b
          + in_channel_size * c;
      float *output_ptr = output + out_batch_num_values * b
          + out_channel_size * c;
      for (index_t y = 0; y < out_height; ++y) {
        const T *ys_input_lower_ptr = input_ptr + ys[y].lower * in_width;
        const T *ys_input_upper_ptr = input_ptr + ys[y].upper * in_width;
        const float ys_lerp = ys[y].lerp;
        for (index_t x = 0; x < out_width; ++x) {
          auto xs_lower = xs[x].lower;
          auto xs_upper = xs[x].upper;
          auto xs_lerp = xs[x].lerp;

          const float top_left = ys_input_lower_ptr[xs_lower];
          const float top_right = ys_input_lower_ptr[xs_upper];
          const float bottom_left = ys_input_upper_ptr[xs_lower];
          const float bottom_right = ys_input_upper_ptr[xs_upper];

          output_ptr[x] =
              ComputeLerp(top_left, top_right, bottom_left, bottom_right,
                                      xs_lerp, ys_lerp);
        }
        output_ptr += out_width;
      }
    }
  }
}
}

template<DeviceType D, typename T>
struct ResizeBilinearFunctor {
  bool align_corners_;

  ResizeBilinearFunctor(bool align_corners)
      : align_corners_(align_corners) {}

  void operator()(const T *input, T *output,
                  index_t n, index_t channels, index_t in_height,
                  index_t in_width, index_t out_height, index_t out_width) {
    if (out_height == in_height && out_width == in_width) {
      std::copy(input, input + channels * in_height * in_width, output);
      return;
    }

    float height_scale =
        CalculateResizeScale(in_height, out_height, align_corners_);
    float
        width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

    std::vector<CachedInterpolation> ys(out_height + 1);
    std::vector<CachedInterpolation> xs(out_width + 1);

    // Compute the cached interpolation weights on the x and y dimensions.
    ComputeInterpolationWeights(out_height, in_height, height_scale, ys.data());
    ComputeInterpolationWeights(out_width, in_width, width_scale, xs.data());

    ResizeImage(input, n, in_height, in_width, out_height, out_width,
                channels, xs, ys, output);
  }
};

} //  namespace kernels
} //  namespace mace

#endif // MACE_KERNELS_RESIZE_BILINEAR_H_
