//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONV_POOL_2D_BASE_H_
#define MACE_OPS_CONV_POOL_2D_BASE_H_

#include "mace/core/operator.h"
#include "mace/kernels/conv_pool_2d_util.h"

namespace mace {

template <DeviceType D, class T>
class ConvPool2dOpBase : public Operator<D, T> {
 public:
  ConvPool2dOpBase(const OperatorDef& op_def, Workspace* ws)
      : Operator<D, T>(op_def, ws),
        strides_(OperatorBase::GetRepeatedArgument<int>("strides")),
        padding_(static_cast<Padding>(OperatorBase::GetSingleArgument<int>(
            "padding", static_cast<int>(SAME)))),
        dilations_(OperatorBase::GetRepeatedArgument<int>("dilations")) {}

  void CalOutputSize(const index_t *input_shape,   // NCHW
                     const index_t *filter_shape,  // OIHW
                     index_t *output_shape) {

    MACE_CHECK(dilations_[0] > 0 && dilations_[1] > 0,
               "Invalid dilations, must >= 1");
    MACE_CHECK((dilations_[0] == 1 || strides_[0] == 1) &&
        (dilations_[1] == 1 || strides_[1] == 1),
               "If dilations > 1, strides should be 1");
    MACE_CHECK_NOTNULL(output_shape);
    /*
    * Convlution/pooling arithmetic:
    * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
    * For details, see https://arxiv.org/pdf/1603.07285.pdf or
    * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    */

    index_t output_height, output_width;

    switch (padding_) {
      case VALID:
        output_height = (input_shape[2] - (filter_shape[2] - 1) * dilations_[0] - 1) / strides_[0] + 1;
        output_width = (input_shape[3] - (filter_shape[3] - 1) * dilations_[1] - 1) / strides_[1] + 1;
        break;
      case SAME:
        output_height = (input_shape[2] - 1) / strides_[0] + 1;
        output_width = (input_shape[3] - 1) / strides_[1] + 1;
        break;
      case FULL:
        output_height = (input_shape[2] + (filter_shape[2] - 1) * dilations_[0] - 1) / strides_[0] + 1;
        output_width = (input_shape[3] + (filter_shape[3] - 1) * dilations_[1] - 1) / strides_[1] + 1;
        break;
      default:
        MACE_CHECK(false, "Unsupported padding type: ", padding_);
    }

    output_shape[0] = input_shape[0];
    output_shape[1] = filter_shape[0];
    output_shape[2] = output_height;
    output_shape[3] = output_width;
  }
 protected:
  std::vector<int> strides_;
  Padding padding_;
  std::vector<int> dilations_;
};

}  // namespace mace

#endif  // MACE_OPS_CONV_POOL_2D_BASE_H_
