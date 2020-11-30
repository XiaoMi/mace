cmake_minimum_required(VERSION 3.13 FATAL_ERROR)

project(micro C CXX ASM)

# CMSIS_5 requires C99
set(CMAKE_C_STANDARD 99)

add_compile_options(
  "-Wall;-Wextra"
  "$<$<COMPILE_LANGUAGE:CXX>:-fno-rtti;-fno-exceptions>"
)

option(MACE_MICRO_ENABLE_CMSIS "Whether to enable cmsis driver" OFF)
option(MACE_MICRO_ENABLE_BFLOAT16    "Whether to enable bfloat16 support"         OFF)
option(MACE_MICRO_ENABLE_TESTS "Whether to enable Mace Micro tests"         OFF)
option(MACE_MICRO_ENABLE_EXAMPLES "Whether to enable Mace Micro examples"         OFF)
option(MACE_MICRO_ENABLE_TOOLS "Whether to enable Mace Micro tools" OFF)
option(MACE_MICRO_ENABLE_XTENSA "Whether to enable xa nnlib engine" OFF)

if(MACE_MICRO_ARM_NONE)
  add_definitions(-DMACE_MICRO_ARM_NONE)
endif()

if(MACE_MICRO_ENABLE_BFLOAT16)
  add_definitions(-DMACE_ENABLE_BFLOAT16)
endif()

if(MACE_MICRO_NDEBUG)
  add_definitions(-DMACE_MICRO_NDEBUG)
endif()

if(MACE_MICRO_GCC_ARM)
  include(cmake/config_gcc_arm.cmake)
endif()

#set CMAKE_BUILD_TYPE default value as Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release"
        CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif()

message(STATUS "BUILD_TYPE: " ${CMAKE_BUILD_TYPE})

set(MACE_MICRO_THIRD_PARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
set(MACE_MICRO_TOOLS_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tools)
set(lib_micro_path ${CMAKE_CURRENT_BINARY_DIR}/libmicro.a)

if(HEXAGON6 AND CMAKE_BUILD_TYPE STREQUAL "RELEASE")
  # Does not work with "-O3"
  add_compile_options(-O2)
endif()


add_library(micro STATIC IMPORTED)

add_subdirectory(include)
add_subdirectory(port)
add_subdirectory(base)
add_subdirectory(model)
add_subdirectory(framework)
add_subdirectory(ops)

# add models files
add_subdirectory(codegen)

set(combine_core_stamp ${CMAKE_CURRENT_BINARY_DIR}/combine_core.stamp)
add_custom_command(
    OUTPUT ${combine_core_stamp}
    COMMAND touch ${combine_core_stamp}
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_base>\""
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_framework>\""
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_model>\""
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_port>\""
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/tools/combine_static_libs.sh ${lib_micro_path} "\"$<TARGET_OBJECTS:micro_ops>\""
    DEPENDS micro_base micro_framework micro_model micro_port micro_ops
    COMMENT "Combining libs ..."
)

add_custom_target(micro_combine_core
  ALL DEPENDS ${combine_core_stamp}
)

set_target_properties(micro PROPERTIES
  IMPORTED_LOCATION ${lib_micro_path}
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include/public
)

add_dependencies(micro micro_combine_core)

install(FILES include/public/micro.h DESTINATION include)
install(FILES ${lib_micro_path} DESTINATION lib)

if(HEXAGON OR HEXAGON_STUB)
  include(cmake/find_hexagon_sdk.cmake)
endif()

if(MACE_MICRO_ENABLE_TOOLS)
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
