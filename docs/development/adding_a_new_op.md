Adding a new Op
===============

You can create a custom op if it is not supported yet.

To add a custom op, you need to follow these steps:

Register the new OpDef information
----------------------------------
Register the OpDef information about which devices the operation could run on.
Registry file is in `mace/ops/ops_def_register.cc`
```c++
#include "mace/ops/ops_def_register.h"

namespace mace {
namespace ops {

void RegisterOpDefs(OpDefRegistryBase *op_def_registry) {
  MACE_REGISTER_OP_DEF(
      op_def_registry,
      OpRegistrationBuilder("MyCustomOp")
          .SetDevicePlaceFunc([]() -> std::vector<DeviceType> {
            return {DeviceType::CPU, DeviceType::GPU};
          }));
  ......
}
}  // namespace ops
}  // namespace mace

```

Implement the Operation
-----------------------
The Best way is to refer to the implementation of other operator(e.g. `/mace/kernels/activation.cc`)

Define the new Op class in `mace/kernels/my_custom_op.cc`.
1. CPU code: just write the code in `mace/kernels/my_custom_op.cc`.
2. GPU code: Kernel API is defined in `mace/kernels/my_custom_op.h`, 
Kernel based on Image is realized in `mace/kernels/opencl/image/my_custom_op.cc`,
Kernel based on Buffer is realized in `mace/kernels/opencl/buffer/my_custom_op.cc`.
 
The structure like the following code.
```c++
#include "mace/core/operator.h"

namespace mace {
namespace kernels {

template <DeviceType D, class T>
class MyCustomOp;

template <>
class MyCustomOp<DeviceType::CPU, float> : public Operation {
...
}

#ifdef MACE_ENABLE_OPENCL
template <typename T>
class ActivationOp<DeviceType::GPU, T> : public Operation {
...
};
#endif  // MACE_ENABLE_OPENCL

}  // namespace ops
}  // namespace mace

```

Register the Operation
-----------------------
1, Add register function in `mace/kernels/my_custom_op.cc`
```c++
#include "mace/core/operator.h"

namespace mace {
namespace kernels {

void RegisterMyCustomOp(OpRegistryBase *op_registry) {
  MACE_REGISTER_OP(op_registry, "MyCustomOp", ActivationOp,
                   DeviceType::CPU, float);

#ifdef MACE_ENABLE_OPENCL
  MACE_REGISTER_OP(op_registry, "MyCustomOp", ActivationOp,
                   DeviceType::GPU, float);

  MACE_REGISTER_OP(op_registry, "MyCustomOp", ActivationOp,
                   DeviceType::GPU, half);
#endif  // MACE_ENABLE_OPENCL
}
}  // namespace ops
}  // namespace mace
```
2, And then register the new Op in `mace/kernels/ops_register.cc`.
```
#include "mace/kernels/ops_register.h"

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
