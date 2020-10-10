cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(micro C CXX ASM)

# CMSIS_5 requires C99
set(CMAKE_C_STANDARD 99)

add_compile_options("-Wall;-Wextra")

option(MACE_MICRO_ENABLE_CMSIS "Whether to enable cmsis driver" OFF)
option(MACE_MICRO_ENABLE_BFLOAT16    "Whether to enable bfloat16 support"         OFF)
option(MACE_MICRO_ENABLE_TESTS "Whether to enable Mace Micro tests"         OFF)
option(MACE_MICRO_ENABLE_EXAMPLES "Whether to enable Mace Micro examples"         OFF)

if(MACE_MICRO_GCC_ARM)
  include(cmake/config_gcc_arm.cmake)
endif()

#set CMAKE_BUILD_TYPE default value as Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

if(MACE_MICRO_ENABLE_CMSIS)
  function(compilerSpecificCompileOptions PROJECTNAME ROOT)
    target_compile_options(${PROJECTNAME}
      PRIVATE "-Wno-unused-parameter"
      PRIVATE "-Wno-sign-compare"
      PRIVATE "-Wno-strict-aliasing"
      PRIVATE "-Wno-unused-variable"
    )
  endfunction()
  set(ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/CMSIS_5)

  include_directories(${ROOT}/CMSIS/Core/Include)

  add_subdirectory(${ROOT}/CMSIS/DSP/Source EXCLUDE_FROM_ALL)
  add_subdirectory(${ROOT}/CMSIS/NN/Source  EXCLUDE_FROM_ALL)

  target_include_directories(CMSISDSP INTERFACE ${ROOT}/CMSIS/Core/Include)
  target_include_directories(CMSISNN INTERFACE ${ROOT}/CMSIS/Core/Include)
  include_directories(third_party/CMSIS_5/CMSIS/Core/Include)
endif()

if(HEXAGON6)
  # Does not work with "-O3"
  add_compile_options(-O2)
endif()

if(MACE_MICRO_ARM_NONE)
  add_definitions(-DMACE_MICRO_ARM_NONE)
endif()

if(MACE_MICRO_ENABLE_BFLOAT16)
  add_definitions(-DMACE_ENABLE_BFLOAT16)
endif()

if(MACE_MICRO_NDEBUG)
  add_definitions(-DMACE_MICRO_NDEBUG)
endif()

add_subdirectory(include)
add_subdirectory(port)
add_subdirectory(base)
add_subdirectory(model)
add_subdirectory(framework)
add_subdirectory(ops)

file(GLOB micro_base_srcs base/*.cc)
file(GLOB micro_codegen_models_srcs codegen/models/**/*.cc)
file(GLOB micro_codegen_engines_srcs codegen/engines/**/*.cc)
file(GLOB micro_framework_srcs framework/*.cc)
file(GLOB micro_models_srcs model/*.cc)
file(GLOB micro_ops_nhwc_base_srcs ops/nhwc/base/*.cc)
file(GLOB micro_ops_nhwc_srcs ops/nhwc/*.cc)
file(GLOB micro_ops_nhwc_cmsis_nn_srcs ops/nhwc/cmsis_nn/*.cc)
file(GLOB micro_ops_srcs ops/*.cc)
file(GLOB micro_ops_utils_srcs ops/utils/*.cc)
file(GLOB micro_port_srcs port/*.cc)

list(APPEND micro_src
  ${micro_base_srcs}
  ${micro_framework_srcs}
  ${micro_models_srcs}
  ${micro_ops_srcs}
  ${micro_ops_nhwc_srcs}
  ${micro_ops_nhwc_base_srcs}
  ${micro_ops_utils_srcs}
  ${micro_port_srcs}
)

if(MACE_MICRO_ENABLE_CMSIS)
  list(APPEND micro_src ${micro_ops_nhwc_cmsis_nn_srcs})
endif()

add_library(micro ${micro_src})
target_include_directories(micro PUBLIC .. PUBLIC include/public)

install(TARGETS micro
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(FILES include/public/micro.h DESTINATION include)

if(MACE_MICRO_ENABLE_CMSIS)
  target_link_libraries(micro PRIVATE CMSISNN)

  install(TARGETS
    CMSISNNReshape
    CMSISNNBasicMaths
    CMSISNNConcatenation
    CMSISNNFullyConnected
    CMSISNNConvolutions
    CMSISNNActivation
    CMSISNNPooling
    CMSISNNSoftmax
    CMSISNNSupport
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
  )
endif()

add_subdirectory(codegen)

if(HEXAGON OR HEXAGON_STUB)
  include(cmake/find_hexagon_sdk.cmake)
endif()

if(NOT HEXAGON AND MICRO_MODEL_NAME)
  add_subdirectory(third_party/gflags EXCLUDE_FROM_ALL)
  add_subdirectory(tools)
endif()

if(MACE_MICRO_ENABLE_TESTS)
  add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  add_subdirectory(test)
endif()

if(MACE_MICRO_ENABLE_EXAMPLES)
  add_subdirectory(examples)
endif()
