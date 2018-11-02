Adding a new Op
===============

You can create a custom op if it is not supported yet.

To add a custom op, you need to follow these steps:

Implement the Operation
-----------------------
The Best way is to refer to the implementation of other operator(e.g. `/mace/ops/activation.cc`)

Define the new Op class in `mace/ops/my_custom_op.cc`.
1. ARM kernels: Kernel about NEON is located at `mace/ops/arm/my_custom_op.cc`
2. GPU kernels: OpenCL kernel API is defined in `mace/ops/opencl/my_custom_op.h`, 
    * Kernel based on Image is realized in `mace/ops/opencl/image/my_custom_op.cc`,
    * Kernel based on Buffer is realized in `mace/ops/opencl/buffer/my_custom_op.cc`.
    * OpenCL kernel file is realized in `mace/ops/opencl/cl/my_custom_op.cl`.
    * Add the path of opencl kernel file in file `mace/repository/opencl-kernel/opencl_kernel_configure.bzl`
 
The structure of Op is like the following code.
```c++
#include "mace/core/operator.h"

namespace mace {
namespace ops {

template <DeviceType D, class T>
class MyCustomOp;

template <>
class MyCustomOp<DeviceType::CPU, float> : public Operation {
...
}

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class MyCustomOp<DeviceType::GPU, T> : public Operation {
...
};
#endif  // MACE_ENABLE_OPENCL

void RegisterMyCustomOp(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "MyCustomOp", MyCustomOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "MyCustomOp", MyCustomOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "MyCustomOp", MyCustomOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}

}  // namespace ops
}  // namespace mace

```

Register the Operation
-----------------------
Register the new Op in `mace/ops/ops_register.cc`.
```
#include "mace/ops/ops_register.h"

namespace mace {
namespace ops {
// Keep in lexicographical order

...

extern void RegisterMyCustomOp(OpRegistryBase *op_registry);

...

}  // namespace ops


OpRegistry::OpRegistry() : OpRegistryBase() {
  // Keep in lexicographical order

  ...

  ops::RegisterMyCustomOp(this);

  ...

}

}  // namespace mace
```
Add UTs
----------------------
Add operation unit tests in `mace/ops/my_custom_op_test.cc`

Add benchmark
----------------------
Add operation benchmark in `mace/ops/my_custom_op_benchmark.cc`
It's strongly recommended to add unit tests and micro benchmarks for your
new Op. If you wish to contribute back, it's required.

Add Op in model converter
-------------------------
You need to add this new Op in the model converter.

Document the new Op
---------------------
Finally, add an entry in operator table in the document.
