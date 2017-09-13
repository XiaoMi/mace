//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/neon/conv_2d_neon_3x3.h"

namespace mace {
namespace kernels {

static inline void ConstructInputWithPadding(const float* input,
                                             const index_t* input_shape,
                                             const int* paddings,
                                             Tensor* output_tensor) {
  index_t batch    = input_shape[0];
  index_t channels = input_shape[1];
  index_t height   = input_shape[2];
  index_t width    = input_shape[3];

  std::vector<index_t> output_shape({batch,
                                     channels,
                                     paddings[0] + height,
                                     paddings[1] + width});

  const index_t output_width = output_shape[3];
  const int padded_top  = paddings[0] / 2;
  const int padded_left = paddings[1] / 2;

  output_tensor->Resize(output_shape);
  float* output_ptr = output_tensor->mutable_data<float>();
  memset(output_ptr, 0, output_tensor->size() * sizeof(float));

  // Skip the padded top rows
  output_ptr += padded_top * output_width;
  for (; batch > 0; --batch) {
    for (; channels > 0; --channels) {
      for(; height > 0; --height) {
        memcpy(output_ptr + padded_left, input, width * sizeof(float));
        input += width;
        output_ptr += output_width;
      }
      // Skip the padded bottom in this channel and top in the next channel
      output_ptr += paddings[0] * output_width;
    }
  }
}

extern void Conv2dNeonK1x1S1(const float* input, const index_t* input_shape,
                             const float* filter, const float* bias,
                             float* output, const index_t* output_shape);


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
  // Selection matrix: kernel_size x stride_size
  static const Conv2dNeonFunction selector[5][2] = {
          {
                  Conv2dNeonK1x1S1,
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
  index_t kernel_h = filter_shape[2];
  index_t kernel_w = filter_shape[3];
  if (kernel_h != kernel_w || kernel_h > 5 ||
      strides_[0] != strides_[1] || strides_[0] > 2 ||
      dilations_[0] != 1 || dilations_[1] != 1 ||
      selector[kernel_h - 1][strides_[0] - 1] == nullptr) {
    LOG(WARNING) << "NEON conv2d kernel not implementated, using slow vesion";
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

  // Keep this alive during kernel execution
  Tensor padded_input;
  if (paddings_[0] > 0 || paddings_[1] > 0) {
    ConstructInputWithPadding(input, input_shape, paddings_, &padded_input);
    input = padded_input.data<float>();
    input_shape = padded_input.shape().data();
  }
  auto conv2d_neon_func = selector[kernel_h - 1][strides_[0] - 1];
  conv2d_neon_func(input,
                   input_shape,
                   filter,
                   bias,
                   output,
                   output_shape);
}

} //  namespace kernels
} //  namespace mace
