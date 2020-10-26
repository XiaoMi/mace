if(NOT ARM_CPU)
  message(FATAL_ERROR "please set ARM_CPU, such as: -DARM_CPU=cortex-m4. We set -mcpu=${ARM_CPU}")
endif()

add_compile_options("-mcpu=${ARM_CPU};-mthumb")
add_compile_options("-ffunction-sections;-fdata-sections")

# floating-point ABI
option(MACE_MICRO_ENABLE_HARDFP "Whether to use hard float-point ABI" ON)

if(MACE_MICRO_ENABLE_HARDFP)
  add_compile_options("-mfloat-abi=hard")
else()
  add_compile_options("-mfloat-abi=softfp")
endif()

# FPU
if (ARM_CPU STREQUAL "cortex-m55" )
  add_compile_options("-mfpu=fpv5-d16")
  add_link_options("-mfpu=fpv5-d16")
endif()

if (ARM_CPU STREQUAL "cortex-m33" )
  add_compile_options("-mfpu=fpv5-sp-d16")
  add_link_options("-mfpu=fpv5-sp-d16")
endif()

if (ARM_CPU STREQUAL "cortex-m7" )
  add_compile_options("-mfpu=fpv5-d16")
  add_link_options("-mfpu=fpv5-d16")
endif()

if (ARM_CPU STREQUAL "cortex-m4" )
  add_compile_options("-mfpu=fpv4-sp-d16")
  add_link_options("-mfpu=fpv4-sp-d16")
endif()
