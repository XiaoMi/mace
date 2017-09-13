//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/neon/conv_2d_neon_3x3.h"

namespace mace {
namespace kernels {

static inline void ConstructInputWithPadding(const float* input, const index_t* input_shape,
                                             const int* paddings,
                                             Tensor& output_tensor,
                                             std::vector<index_t>& output_shape) {
  index_t batch    = input_shape[0];
  index_t channels = input_shape[1];
  index_t height   = input_shape[2];
  index_t width    = input_shape[3];
  output_shape[0] = batch;
  output_shape[1] = channels;
  output_shape[2] = paddings[0] + height;
  output_shape[3] = paddings[1] + width;
  index_t output_width = output_shape[3];

  int padded_left   = paddings[1] / 2;

  output_tensor.Resize(output_shape);
  float* output_ptr = output_tensor.mutable_data<float>();
  memset(output_ptr, 0, output_tensor.size() * sizeof(float));
  output_ptr += paddings[0] / 2 * output_width;

  for (; batch > 0; --batch) {
    for (; channels > 0; --channels) {
      for(; height > 0; --height) {
        memcpy(output_ptr+padded_left, input, width*sizeof(float));
        input += width;
        output_ptr += output_width;
      }
      output_ptr += paddings[0] * output_width;
    }
  }
}

template<>
void Conv2dFunctor<DeviceType::NEON, float>::operator()(const float* input, // NCHW
                                                        const index_t* input_shape,
                                                        const float* filter, // c_out, c_in, kernel_h, kernel_w
                                                        const index_t* filter_shape,
                                                        const float* bias, // c_out
                                                        float* output, // NCHW
                                                        const index_t* output_shape) {

  typedef void (*Conv2dNeonFunction)(const float* input, // NCHW
                           const index_t* input_shape,
                           const float* filter, // c_out, c_in, kernel_h, kernel_w
                           const float* bias, // c_out
                           float* output, // NCHW
                           const index_t* output_shape);
  static const Conv2dNeonFunction selector[5][2] = {
          {
                  nullptr,
                  nullptr
          },
          {
                  nullptr,
                  nullptr
          },
          {
                  Conv2dNeonK3x3S1,
                  nullptr
          },
          {
                  nullptr,
                  nullptr
          },
          {
                  nullptr,
                  nullptr
          }
  };
  // not implement yet
  if (paddings_[0] != paddings_[1] || paddings_[0] > 5 ||
          strides_[0] != strides_[1] || strides_[0] > 4 ||
          dilations_[0] != 1 || dilations_[1] != 1 ||
          selector[paddings_[0]-1][strides_[0]-1] == nullptr) {
    Conv2dFunctor<DeviceType::CPU, float>(strides_, paddings_, dilations_)(
            input,
            input_shape,
            filter,
            filter_shape,
            bias,
            output,
            output_shape
    );
  }
  Tensor padded_input;
  std::vector<index_t> padded_input_shape(4);
  ConstructInputWithPadding(input, input_shape, paddings_, padded_input, padded_input_shape);
  auto conv2d_neon_func = selector[paddings_[0] - 1][strides_[0] - 1];
  conv2d_neon_func(
          padded_input.data<float>(),
          padded_input_shape.data(),
          filter,
          bias,
          output,
          output_shape
  );
}

} //  namespace kernels
} //  namespace mace