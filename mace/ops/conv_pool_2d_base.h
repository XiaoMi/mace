//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONV_POOL_2D_BASE_H_
#define MACE_OPS_CONV_POOL_2D_BASE_H_

#include "mace/core/operator.h"

namespace mace {

template<DeviceType D, class T>
class ConvPool2dOpBase : public Operator<D, T> {
 public:
  ConvPool2dOpBase(const OperatorDef &op_def, Workspace *ws)
    : Operator<D, T>(op_def, ws),
    strides_(OperatorBase::GetRepeatedArgument<int>("strides")),
    padding_(static_cast<Padding>(
          OperatorBase::GetSingleArgument<int>("padding",
                                               static_cast<int>(SAME)))),
    dilations_(OperatorBase::GetRepeatedArgument<int>("dilations")) {}

  void CalcPaddingAndOutputSize(const Tensor* input,
                                const Tensor* filter,
                                std::vector<index_t>* output_shape,
                                std::vector<int>* padding_size) {
    MACE_CHECK(dilations_[0] > 0 && dilations_[1] > 0,
        "Invalid dilations, must >= 1");
    /*
     * Convlution/pooling arithmetic:
     * o = (i + 2 * p - k - (k - 1) * (d - 1)) / s + 1
     * For details, see https://arxiv.org/pdf/1603.07285.pdf or
     * http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
     */
    auto& input_shape = input->shape();
    auto& filter_shape = filter->shape(); // HWIO
    int kernel_h = filter_shape[0];
    int kernel_w = filter_shape[1];
    int output_channel = filter_shape[3];
    MACE_CHECK(input_shape[1] == filter_shape[2],
        input_shape[1], " != ", filter_shape[2]);

    *padding_size = {0, 0};
    switch (padding_) {
      case VALID:
        break;
      case SAME:
        (*padding_size)[0] = kernel_h / 2; 
        (*padding_size)[1] = kernel_w / 2;
        break;
      case FULL:
        (*padding_size)[0] = kernel_h - 1;
        (*padding_size)[1] = kernel_w - 1;
        break;
      default:
        MACE_CHECK(false, "Unsupported padding type: ", padding_);
    }
    *output_shape = std::vector<index_t>(4); // NCHW
    (*output_shape)[0] = input_shape[0];
    (*output_shape)[1] = output_channel;
    (*output_shape)[2] = (input_shape[2] + 2 * (*padding_size)[0] - kernel_h -
                          (kernel_h - 1) * (dilations_[0] - 1)) /
                         strides_[0] + 1;
    (*output_shape)[3] = (input_shape[3] + 2 * (*padding_size)[1] - kernel_w -
                          (kernel_w - 1) * (dilations_[1] - 1)) /
                         strides_[1] + 1;
  }

  enum Padding {
    VALID = 0, // No padding
    SAME  = 1, // Pads with half the filter size (rounded down) on both sides
    FULL  = 2, // Pads with one less than the filter size on both sides
  };

 protected:
  std::vector<int> strides_;
  Padding padding_;
  std::vector<int> dilations_;
};

} // namespace mace

#endif // MACE_OPS_CONV_POOL_2D_BASE_H_
