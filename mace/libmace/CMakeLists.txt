set(LIBMACE_SRCS capability.cc mace.cc)

add_library(mace SHARED ${LIBMACE_SRCS})
target_link_libraries(mace ops)
if(NOT APPLE)
  set_target_properties(mace PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mace_version_script.lds")
endif(NOT APPLE)

add_library(mace_static STATIC ${LIBMACE_SRCS})
target_link_libraries(mace_static ops)

install(TARGETS mace LIBRARY DESTINATION lib)
install(TARGETS mace_static ARCHIVE DESTINATION lib)
