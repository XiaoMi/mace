//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#include <arm_neon.h>
#include "mace/kernels/conv_2d.h"
#include "mace/kernels/neon/conv_2d_neon_base.h"

namespace mace {
namespace kernels {

static inline void ConstructInputWithPadding(const float* input, const index_t* input_shape,
                                             const int* padding,
                                             std::unique_ptr<float>& output,
                                             index_t* output_shape) {

}

template<>
void Conv2dFunctor<DeviceType::NEON, float>::operator()(const float* input, // NCHW
                                                        const index_t* input_shape,
                                                        const float* filter, // c_out, c_in, kernel_h, kernel_w
                                                        const index_t* filter_shape,
                                                        const float* bias, // c_out
                                                        float* output, // NCHW
                                                        const index_t* output_shape) {

  static const bool selector[5][4] = {
          {true, false, false, false},
          {false, false, false, false},
          {true, true, false, false},
          {false, false, false, false},
          {true, false, false, false},
  };
  // not implement yet
  if (paddings_[0] != paddings_[1] || paddings_[0] > 5 ||
          strides_[0] != strides_[1] || strides_[0] > 4 ||
          dilations_[0] != 1 || dilations_[1] != 1 ||
          !selector[paddings_[0]-1, strides_[0]-1]) {
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
  std::unique_ptr<float> padded_input;
  index_t padded_input_shape[4];
  ConstructInputWithPadding(input, input_shape, paddings_, padded_input, padded_input_shape);
  Conv2dNeon<paddings_[0], paddings_[1], strides_[0], strides_[1]>(
          padded_input.get(),
          padded_input_shape,
          filter,
          bias,
          output,
          output_shape
  );
}

} //  namespace kernels
} //  namespace mace