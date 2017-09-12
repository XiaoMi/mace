//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONV_POOL_2D_BASE_H_
#define MACE_OPS_CONV_POOL_2D_BASE_H_

#include "mace/core/operator.h"

namespace mace {

enum Padding {
  VALID = 0, // No padding
  SAME  = 1, // Pads with half the filter size (rounded down) on both sides
  FULL  = 2, // Pads with one less than the filter size on both sides
};

template<DeviceType D, class T>
class ConvPool2dOpBase : public Operator<D, T> {
 public:
  ConvPool2dOpBase(const OperatorDef& op_def, Workspace* ws)
    : Operator<D, T>(op_def, ws),
    strides_(OperatorBase::GetRepeatedArgument<int>("strides")),
    padding_(static_cast<Padding>(
          OperatorBase::GetSingleArgument<int>("padding",
                                               static_cast<int>(SAME)))),
    dilations_(OperatorBase::GetRepeatedArgument<int>("dilations")) {}

  void CalcPaddingAndOutputSize(const index_t* input_shape,  // NCHW
                                const index_t* filter_shape,  // OIHW
                                std::vector<index_t>* output_shape,
                                std::vector<int>* padding_size) {
    MACE_CHECK(dilations_[0] > 0 && dilations_[1] > 0,
               "Invalid dilations, must >= 1");
    MACE_CHECK((dilations_[0] == 1 || strides_[0] == 1) &&
               (dilations_[1] == 1 || strides_[1] == 1),
               "If dilations > 1, strides should be 1");
    /*
    * Convlution/pooling arithmetic:
    * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
    * For details, see https://arxiv.org/pdf/1603.07285.pdf or
    * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    */
    *padding_size = {0, 0};

    index_t output_height, output_width;
    index_t kernel_height = filter_shape[2];
    index_t kernel_width = filter_shape[3];
    index_t output_channels = filter_shape[0];

    index_t k_extent_height = (kernel_height - 1) * dilations_[0] + 1;
    index_t k_extent_width = (kernel_width - 1) * dilations_[1] + 1;

    switch (padding_) {
      case VALID:
        output_height = (input_shape[2] - k_extent_height) / strides_[0] + 1;
        output_width = (input_shape[3] - k_extent_width) / strides_[1] + 1;
        break;
      case SAME:
        output_height = (input_shape[2] - 1) / strides_[0] + 1;
        output_width = (input_shape[3] - 1) / strides_[1] + 1;
        break;
      case FULL:
        output_height = (input_shape[2] + k_extent_height - 2) / strides_[0] + 1;
        output_width = (input_shape[3] + k_extent_width - 2) / strides_[1] + 1;
        break;
      default:
        MACE_CHECK(false, "Unsupported padding type: ", this->padding_);
    }

    // Note: TensorFlow may padded one more on the right/bottom side
    // TODO may be it's better to also truncate the left/top to
    // utilize the more centered features. We need to benchmark
    // based on the model accuracy.

    (*padding_size)[0] = (output_height - 1) * strides_[0] +
                         k_extent_height - input_shape[2];
    (*padding_size)[1] = (output_width - 1) * strides_[1] +
                         k_extent_width - input_shape[3];

    *output_shape = std::vector<index_t>(4); // NCHW
    (*output_shape)[0] = input_shape[0];
    (*output_shape)[1] = output_channels;
    (*output_shape)[2] = output_height;
    (*output_shape)[3] = output_width;
  }

 protected:
  std::vector<int> strides_;
  Padding padding_;
  std::vector<int> dilations_;
};

} // namespace mace

#endif // MACE_OPS_CONV_POOL_2D_BASE_H_
