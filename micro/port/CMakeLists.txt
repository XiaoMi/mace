add_library(micro_port api.cc)
target_include_directories(micro_port
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
  PUBLIC $<INSTALL_INTERFACE:include>
)

if(HEXAGON OR HEXAGON_STUB)
  target_link_libraries(micro_port  hexagon_sdk_headers_dsp)
endif()
