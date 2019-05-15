# Generate MACE version source
set(MACE_VERSION_GEN_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/tools/gen_version_source.sh")
set(MACE_VERSION_SRC "${CMAKE_CURRENT_BINARY_DIR}/version/version.cc")
add_custom_command(
  OUTPUT ${MACE_VERSION_SRC}
  COMMAND ${MACE_VERSION_GEN_TOOL} ${MACE_VERSION_SRC}
  DEPENDS ${MACE_VERSION_GEN_TOOL}
  COMMENT "Generating version source file" )

add_custom_target(version_src DEPENDS ${MACE_VERSION_SRC}
  COMMENT "Checking if re-generation is required" )

add_library(generated_version ${MACE_VERSION_SRC})
add_dependencies(generated_version version_src)

# Generate OpenCL kernel source
set(MACE_OPENCL_KERNELS_GEN_TOOL "${PROJECT_SOURCE_DIR}/mace/python/tools/encrypt_opencl_codegen.py")
set(MACE_OPENCL_KERNELS_DIR "${PROJECT_SOURCE_DIR}/mace/ops/opencl/cl")
set(MACE_OPENCL_KERNELS_SRC "${CMAKE_CURRENT_BINARY_DIR}/opencl/encrypt_opencl_kernel.cc")
add_custom_command(
  OUTPUT ${MACE_OPENCL_KERNELS_SRC}
  COMMAND python ${MACE_OPENCL_KERNELS_GEN_TOOL} --output_path=${MACE_OPENCL_KERNELS_SRC} --cl_kernel_dir=${MACE_OPENCL_KERNELS_DIR}
  DEPENDS ${MACE_OPENCL_KERNELS_GEN_TOOL}
  COMMENT "Generating OpenCL kernel source file" )

add_custom_target(opencl_kernel_src DEPENDS ${MACE_OPENCL_KERNELS_SRC}
  COMMENT "Checking if re-generation is required" )

add_library(generated_opencl_kernel ${MACE_OPENCL_KERNELS_SRC})
add_dependencies(generated_opencl_kernel opencl_kernel_src)

install(TARGETS generated_version ARCHIVE DESTINATION lib)
install(TARGETS generated_opencl_kernel ARCHIVE DESTINATION lib)
