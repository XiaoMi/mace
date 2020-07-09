cmake_minimum_required(VERSION 3.7 FATAL_ERROR)
message("CMAKE_VERSION: ${CMAKE_VERSION}")
project(micro C CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

if(HEXAGON6)
  # Does not work with "-O3"
  add_compile_options(-O2)
endif()

if(MACE_MICRO_ARM_NONE)
  add_definitions(-DMACE_MICRO_ARM_NONE)
endif()

option(MACE_ENABLE_BFLOAT16    "Whether to enable bfloat16 support"         OFF)
option(MACE_MICRO_ENABLE_TESTS "Whether to enable Mace Micro tests"         ON)

if(MACE_ENABLE_BFLOAT16)
  add_definitions(-DMACE_ENABLE_BFLOAT16)
endif()

if(MACE_MICRO_NDEBUG)
  add_definitions(-DMACE_MICRO_NDEBUG)
endif()

include(third_party/third_party.cmake)

add_subdirectory(include)
add_subdirectory(port)
add_subdirectory(base)
add_subdirectory(model)
add_subdirectory(framework)
add_subdirectory(ops)
add_subdirectory(codegen)

file(GLOB micro_base_srcs base/*.cc)
file(GLOB micro_codegen_models_srcs codegen/models/**/*.cc)
file(GLOB micro_codegen_engines_srcs codegen/engines/**/*.cc)
file(GLOB micro_framework_srcs framework/*.cc)
file(GLOB micro_models_srcs model/*.cc)
file(GLOB micro_ops_nhwc_base_srcs ops/nhwc/base/*.cc)
file(GLOB micro_ops_nhwc_srcs ops/nhwc/*.cc)
file(GLOB micro_ops_srcs ops/*.cc)
file(GLOB micro_ops_utils_srcs ops/utils/*.cc)
file(GLOB micro_port_srcs port/*.cc)

# To build a single library
add_library(micro
  ${micro_base_srcs}
  ${micro_codegen_models_srcs}
  ${micro_codegen_engines_srcs}
  ${micro_framework_srcs}
  ${micro_models_srcs}
  ${micro_ops_srcs}
  ${micro_ops_nhwc_srcs}
  ${micro_ops_nhwc_base_srcs}
  ${micro_ops_utils_srcs}
  ${micro_port_srcs}
)
target_include_directories(micro PUBLIC ..)

install(TARGETS micro
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

if(HEXAGON OR HEXAGON_STUB)
  include(cmake/find_hexagon_sdk.cmake)
endif()

if(NOT HEXAGON)
  add_subdirectory(tools)
endif()

if(MACE_MICRO_ENABLE_TESTS)
  add_subdirectory(test)
endif(MACE_MICRO_ENABLE_TESTS)
