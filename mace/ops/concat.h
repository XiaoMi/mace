//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONCAT_H_
#define MACE_OPS_CONCAT_H_

#include "mace/core/operator.h"
#include "mace/kernels/concat.h"
#include "mace/proto/mace.pb.h"
namespace mace {

template <DeviceType D, typename T>
class ConcatOp : public Operator<D, T> {
 public:
  ConcatOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws) {}

  bool Run() override {
    const int32_t inputs_count = this->InputSize() - 1;
    const std::vector<const Tensor *> input_list = this->Inputs();
    const Tensor *axis_tensor = this->Input(inputs_count);
    MACE_CHECK(axis_tensor->dim_size() == 0,
               "axis should be a scalar integer, but got shape: ",
               axis_tensor->dim_size());
    Tensor::MappingGuard axis_mapper(axis_tensor);
    const int32_t concat_axis = *(axis_tensor->data<int32_t>());
    const int32_t input_dims = input_list[0]->dim_size();
    const int32_t axis =
        concat_axis < 0 ? concat_axis + input_dims : concat_axis;
    MACE_CHECK((0 <= axis && axis < input_dims),
               "Expected concatenating axis in the range [", -input_dims, ", ",
               input_dims, "], but got", concat_axis);

    Tensor *output = this->Output(OUTPUT);

    functor_(input_list, axis, output);
    return true;
  }

 private:
  kernels::ConcatFunctor<D, T> functor_;

 private:
  OP_OUTPUT_TAGS(OUTPUT);
};

}  //  namespace mace

#endif  //  MACE_OPS_CONCAT_H_
