add_library(micro_port api.cc)
target_include_directories(micro_port
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
  PUBLIC $<INSTALL_INTERFACE:include>
)
target_compile_options(micro_port PRIVATE "-fPIC")
