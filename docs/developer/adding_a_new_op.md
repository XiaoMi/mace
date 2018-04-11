Adding a New Op
===============

You can create a custum op if it is not covered by existing mace library.

To add a custom op in mace you'll need to:

- Define the new op class and Registry function in op_name.h and op_name.cc under mace/ops directory. 


```
op_name.cc  

#include "mace/ops/op_name.h"

namespace mace {
namespace ops {

void Register_Custom_Op(OperatorRegistry *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("op_name")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    Custom_Op<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("op_name")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    Custom_Op<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("op_name")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    Custom_Op<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace

```


```
op_name.h

#ifndef MACE_OPS_CUSTOM_OP_H_
#define MACE_OPS_CUSTOM_OP_H_

#include "mace/core/operator.h"
#include "mace/kernels/custom_op.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class CustomOp : public Operator<D, T> {
 public:
  CustomOp(const OperatorDef &op_def, Workspace *ws)
      : Operator<D, T>(op_def, ws),
        functor_() {}

  bool Run(StatsFuture *future) override {
    const Tensor *input = this->Input(INPUT);
    Tensor *output = this->Output(OUTPUT);
   
    functor_(input, output, future);
    return true;
  }

 protected:
  OP_INPUT_TAGS(INPUT);
  OP_OUTPUT_TAGS(OUTPUT);

 private:
  kernels::CustomOpFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_CUSTOM_OP_H_

```

- To Add the new op. You need to implement the cpu version operation in a .h file and gpu version in op_name_opencl.cc and op_name.cl files under the mace/kernels directory.

- Register the new op in core/operator.cc
 
 
- Test and Benchmark
 
    Add an op_name_test.cc file to test all functions of your new op both in cpu and gpu and make sure the new op works fine.
 
     Add an op_benchmark.cc file to benchmark all functions of your new op both in cpu or gpu.
  
 
 
 
