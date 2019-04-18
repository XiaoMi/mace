file(GLOB OPS_COMMON_SRCS common/*.cc)
file(GLOB OPS_REF_KERNELS_SRCS ref/*.cc)

file(GLOB OPS_ARM_NEON_FP32_KERNELS_SRCS
  arm/fp32/*.cc
)
file(GLOB OPS_ARM_NEON_Q8_KERNELS_SRCS
  arm/q8/*.cc
)

file(GLOB OPS_OPENCL_KERNELS_SRCS
  opencl/*.cc
  opencl/cl/*.cc
  opencl/buffer/*.cc
  opencl/image/*.cc
)

file(GLOB OPS_INTERNAL_OPS_SRCS *.cc)

set(OPS_SRCS registry/ops_registry.cc)
set(OPS_SRCS ${OPS_SRCS} ${OPS_COMMON_SRCS})
set(OPS_SRCS ${OPS_SRCS} ${OPS_INTERNAL_OPS_SRCS})
# TODO we need to remove this in production build
set(OPS_SRCS ${OPS_SRCS} ${OPS_REF_KERNELS_SRCS}) 

if(MACE_ENABLE_NEON)
  set(OPS_SRCS ${OPS_SRCS} ${OPS_ARM_NEON_FP32_KERNELS_SRCS})
endif(MACE_ENABLE_NEON)

if(MACE_ENABLE_QUANTIZE)
  set(OPS_SRCS ${OPS_SRCS} ${OPS_ARM_NEON_Q8_KERNELS_SRCS})
endif(MACE_ENABLE_QUANTIZE)

if(MACE_ENABLE_OPENCL)
  set(OPS_SRCS ${OPS_SRCS} ${OPS_OPENCL_KERNELS_SRCS})
endif(MACE_ENABLE_OPENCL)

add_library(ops STATIC ${OPS_SRCS})
target_link_libraries(ops PRIVATE core proto utils port)

if(MACE_ENABLE_QUANTIZE)
  add_dependencies(ops tflite)
endif(MACE_ENABLE_QUANTIZE)

install(TARGETS ops ARCHIVE DESTINATION lib)
