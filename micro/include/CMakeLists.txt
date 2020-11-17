add_library(micro_include INTERFACE)
target_include_directories(micro_include
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR};public>
  INTERFACE $<INSTALL_INTERFACE:include>
)
