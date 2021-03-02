set(OPENCL_SRCS
  core/opencl_executor.cc
  core/opencl_util.cc
  core/opencl_helper.cc
  core/opencl_context.cc
  core/opencl_wrapper.cc
  opencl_buffer_allocator.cc
  opencl_image_allocator.cc
  opencl_image_manager.cc
  opencl_ref_runtime.cc
  opencl_runtime.cc
  transform/image_to_buffer.cc
  transform/buffer_transform.cc
  transform/buffer_to_image.cc
  transform/buffer_type_transform.cc
  transform/buffer_transformer.cc
)

if(MACE_ENABLE_RPCMEM)
  set(OPENCL_SRCS
    ${OPENCL_SRCS}
    qc_ion/opencl_image_qc_ion_allocator.cc
    qc_ion/opencl_base_qc_ion_allocator.cc
    qc_ion/opencl_qc_ion_runtime.cc
    qc_ion/opencl_buffer_qc_ion_allocator.cc
    qc_ion/opencl_qc_ion_executor.cc
  )
endif(MACE_ENABLE_RPCMEM)

add_library(opencl_runtime STATIC ${OPENCL_SRCS})
add_dependencies(opencl_runtime opencl_clhpp)

set(EXTRA_LINK_LIBS ${EXTRA_LINK_LIBS} generated_opencl_kernel dl)
target_link_libraries(opencl_runtime core ${EXTRA_LINK_LIBS})
install(TARGETS opencl_runtime ARCHIVE DESTINATION lib)
