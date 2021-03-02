set(HEXAGON_SRCS
  hexagon_ref_flow.cc
)

add_library(hexagon_flow STATIC ${HEXAGON_SRCS})
target_link_libraries(hexagon_flow core)
install(TARGETS hexagon_flow ARCHIVE DESTINATION lib)
