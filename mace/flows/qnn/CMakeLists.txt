set(QNN_SRCS
  qnn_ref_flow.cc
)

add_library(qnn_flow STATIC ${QNN_SRCS})
target_link_libraries(qnn_flow core)
install(TARGETS qnn_flow ARCHIVE DESTINATION lib)
