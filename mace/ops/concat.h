//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONCAT_H_
#define MACE_OPS_CONCAT_H_

#include "mace/core/operator.h"
#include "mace/kernels/concat.h"
namespace mace {

template <DeviceType D, typename T>
class ConcatOp : public Operator<D, T> {
 public:
  ConcatOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_(OperatorBase::GetSingleArgument<int>("axis", 3)){}

  bool Run(StatsFuture *future) override {
    MACE_CHECK(this->InputSize() >= 2) << "There must be at least two inputs to concat";
    const std::vector<const Tensor *> input_list = this->Inputs();
    const int32_t concat_axis = OperatorBase::GetSingleArgument<int>("axis", 3);
    const int32_t input_dims = input_list[0]->dim_size();
    const int32_t axis =
        concat_axis < 0 ? concat_axis + input_dims : concat_axis;
    MACE_CHECK((0 <= axis && axis < input_dims),
               "Expected concatenating axis in the range [", -input_dims, ", ",
               input_dims, "], but got", concat_axis);

    Tensor *output = this->Output(OUTPUT);

    functor_(input_list, output, future);
    return true;
  }

 private:
  kernels::ConcatFunctor<D, T> functor_;

 private:
  OP_OUTPUT_TAGS(OUTPUT);
};

}  //  namespace mace

#endif  //  MACE_OPS_CONCAT_H_
