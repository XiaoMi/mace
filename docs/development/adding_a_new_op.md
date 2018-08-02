Adding a new Op
===============

You can create a custom op if it is not supported yet.

To add a custom op, you need to follow these steps:

Define the Op class
--------------------
Define the new Op class in `mace/ops/my_custom_op.h`.

```c++
#ifndef MACE_OPS_MY_CUSTOM_OP_H_
#define MACE_OPS_MY_CUSTOM_OP_H_

#include "mace/core/operator.h"
#include "mace/kernels/my_custom_op.h"

namespace mace {
namespace ops {

template <DeviceType D, typename T>
class MyCustomOp : public Operator<D, T> {
 public:
  MyCustomOp(const OperatorDef &op_def, Workspace *ws)
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
  kernels::MyCustomOpFunctor<D, T> functor_;
};

}  // namespace ops
}  // namespace mace

#endif  // MACE_OPS_MY_CUSTOM_OP_H_

```

Register the new Op
--------------------
Define the Ops registering function in `mace/ops/my_custom_op.cc`.
```c++
#include "mace/ops/my_custom_op.h"

namespace mace {
namespace ops {

void Register_My_Custom_Op(OperatorRegistryBase *op_registry) {
  REGISTER_OPERATOR(op_registry, OpKeyBuilder("my_custom_op")
                                     .Device(DeviceType::CPU)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    Custom_Op<DeviceType::CPU, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("my_custom_op")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<float>("T")
                                     .Build(),
                    Custom_Op<DeviceType::OPENCL, float>);

  REGISTER_OPERATOR(op_registry, OpKeyBuilder("my_custom_op")
                                     .Device(DeviceType::OPENCL)
                                     .TypeConstraint<half>("T")
                                     .Build(),
                    Custom_Op<DeviceType::OPENCL, half>);
}

}  // namespace ops
}  // namespace mace

```
And then register the new Op in `mace/ops/ops_register.cc`.
```
#include "mace/ops/ops_register.h"

namespace mace {

namespace ops {
// Keep in lexicographical order

...

extern void Register_My_Custom_Op(OperatorRegistryBase *op_registry);

...

}  // namespace ops


OperatorRegistry::OperatorRegistry() : OperatorRegistryBase() {
  // Keep in lexicographical order

  ...

  ops::Register_My_Custom_Op(this);

  ...

}

}  // namespace mace
```

Implement the Op kernel code
----------------------------
You need to implement the CPU kernel in a `mace/kernels/my_custom_op.h` and
optionally OpenCL kernel in `mace/kernels/kernels/my_custom_op_opencl.cc` and
`mace/kernels/kernels/cl/my_custom_op.cl`. You can also optimize the CPU
kernel with NEON.

Add test and benchmark
----------------------
It's strongly recommended to add unit tests and micro benchmarks for your
new Op. If you wish to contribute back, it's required.

Add Op in model converter
-------------------------
You need to add this new Op in the model converter.

Document the new Op
---------------------
Finally, add an entry in operator table in the document.
