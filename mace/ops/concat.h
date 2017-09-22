//
// Copyright (c) 2017 XiaoMi All rights reserved.
//

#ifndef MACE_OPS_CONCAT_H_
#define MACE_OPS_CONCAT_H_

#include "mace/proto/mace.pb.h"
#include "mace/core/operator.h"
#include "mace/kernels/concat.h"
namespace mace {

template<DeviceType D, typename T>
class ConcatOp : public Operator<D, T> {
 public:
  ConcatOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws) {}

  bool Run() override {
    int32_t values_count = this->InputSize() - 1;
    const Tensor *input0 = this->Input(0);
    const Tensor *axis_tensor = this->Input(values_count);
    MACE_CHECK(axis_tensor->dim_size() == 0,
               "axis should be a scalar integer, but got shape: ",
               axis_tensor->dim_size());
    const int32_t concat_axis = *(axis_tensor->data<int32_t>());
    const int32_t input_dims = input0->dim_size();
    const int32_t axis = concat_axis < 0 ? concat_axis + input_dims : concat_axis;
    MACE_CHECK((0 <= axis && axis < input_dims), "Expected concatenating axis in the range [",
               -input_dims, ", ", input_dims, "], but got", concat_axis);
    std::vector<index_t> output_shape(input0->shape());
    index_t inner_size = 1;
    for (int i = 0; i < axis; ++i) {
      inner_size *= output_shape[i];
    }
    std::vector<index_t> outer_sizes(values_count, 0);
    std::vector<const T *> input_list(values_count, nullptr);
    input_list[0] = input0->data<T>();
    outer_sizes[0] = input0->size() / inner_size;
    const Tensor *input = nullptr;
    for (int i = 1; i < values_count; ++i) {
      input = this->Input(i);
      MACE_CHECK(input->dim_size() == input0->dim_size(), "Ranks of all input tensors must be same.");
      for (int j = 0; j < axis_tensor->dim_size(); ++j) {
        if (j == axis) { continue; }
        MACE_CHECK(input->dim(j) == input0->dim(j), "Dimensions of inputs should equal except axis.");
      }
      input_list[i] = input->data<T>();
      outer_sizes[i] = input->size() / inner_size;
      output_shape[axis] += input->dim(axis);
    }

    Tensor *output = this->Output(OUTPUT);
    output->Resize(output_shape);

    functor_(input_list, inner_size, outer_sizes.data(), output->mutable_data<T>());
    return true;
  }
 private:
  kernels::ConcatFunctor<D, T> functor_;

 private:
  OP_OUTPUT_TAGS(OUTPUT);
};

} //  namespace mace

#endif //  MACE_OPS_CONCAT_H_
