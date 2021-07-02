file(GLOB OPS_SRCS
  ops/*.cc
)

set(QNN_SRCS
  qnn_performance.cc
  op_builder.cc
  qnn_runtime.cc
  qnn_wrapper.cc
)
set(QNN_SRCS ${QNN_SRCS} ${OPS_SRCS})

add_library(qnn_runtime STATIC ${QNN_SRCS})
target_link_libraries(qnn_runtime core)
install(TARGETS qnn_runtime ARCHIVE DESTINATION lib)
