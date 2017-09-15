//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_RESIZE_BILINEAR_H
#define MACE_RESIZE_BILINEAR_H

#include "mace/core/operator.h"
#include "mace/kernels/resize_bilinear.h"

namespace mace {

template <DeviceType D, class T>
class ResizeBilinearOp : public Operator<D, T> {
 public:
  ResizeBilinearOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<D, T>(operator_def, ws),
        functor_(
            OperatorBase::GetSingleArgument<bool>("align_corners", false)) {}

  bool Run() override {
    const Tensor* input = this->Input(0);
    const Tensor* resize_dims = this->Input(1);

    MACE_CHECK(input->dim_size() == 4, "input must be 4-dimensional.",
               input->dim_size());
    MACE_CHECK(resize_dims->dim_size() == 1,
               "resize dim must be 2-dimensional.", resize_dims->dim_size());

    Tensor* output = this->Output(0);

    index_t n = input->dim(0);
    index_t channels = input->dim(1);
    index_t in_height = input->dim(2);
    index_t in_width = input->dim(3);
    index_t out_height = resize_dims->data<index_t>()[0];
    index_t out_width = resize_dims->data<index_t>()[1];
    vector<index_t> out_shape{n, channels, out_height, out_width};
    output->Resize(out_shape);

    const T* input_ptr = input->data<T>();
    T* output_ptr = output->mutable_data<T>();

    functor_(input_ptr, output_ptr, n, channels, in_height, in_width,
             out_height, out_width);
    return true;
  }

 private:
  kernels::ResizeBilinearFunctor<D, T> functor_;
};

}  //  namespace mace

#endif  // MACE_RESIZE_BILINEAR_H
