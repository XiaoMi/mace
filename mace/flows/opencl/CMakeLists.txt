set(OPENCL_SRCS
  opencl_ref_flow.cc
)

add_library(opencl_flow STATIC ${OPENCL_SRCS})
add_dependencies(opencl_flow opencl_clhpp)

target_link_libraries(opencl_flow core ${EXTRA_LINK_LIBS})
install(TARGETS opencl_flow ARCHIVE DESTINATION lib)
