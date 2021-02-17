set(LIBMACE_SRCS capability.cc mace.cc)

add_library(mace SHARED ${LIBMACE_SRCS})
target_link_libraries(mace ops)
if(NOT APPLE)
  set_target_properties(mace PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mace_version_script.lds")
endif(NOT APPLE)

add_library(mace_API STATIC ${LIBMACE_SRCS})
target_link_libraries(mace_API ops)

set(BASE_PATH ${CMAKE_BINARY_DIR})
if(ANDROID)
  set(PORT_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/port/posix/libport_posix.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux_base/libport_linux_base.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/android/libport_android.a")
elseif(APPLE)
  set(PORT_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/port/posix/libport_posix.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/darwin/libport_darwin.a")
elseif(WIN32)
  set(PORT_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/port/windows/libport_windows.a")
else(WIN32)
  set(PORT_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/port/posix/libport_posix.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux_base/libport_linux_base.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux/libport_linux.a")
endif(ANDROID)

string(CONCAT PORT_LIBS_MRI ${PORT_LIBS} "")
configure_file("${CMAKE_SOURCE_DIR}/mace/libmace/mace.mri.in" 
               "${CMAKE_BINARY_DIR}/mace/libmace/mace.mri")

set(STATIC_LIB ${CMAKE_BINARY_DIR}/mace/libmace/libmace_static.a)
add_custom_target(combined ALL
        COMMAND ar -M < ${CMAKE_BINARY_DIR}/mace/libmace/mace.mri
        DEPENDS mace_API ops core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base
        )

foreach(PORT_LIB ${PORT_LIBS})
  string(REPLACE "/" ";" PORT_LIST ${PORT_LIB})
  list(GET PORT_LIST -1 LIB_NAME)
  get_filename_component(LIB_NAME ${LIB_NAME} NAME_WE)
  string(SUBSTRING ${LIB_NAME} 3 -1 LIB_NAME)
  add_dependencies(combined ${LIB_NAME})
endforeach()

add_library(mace_static STATIC IMPORTED GLOBAL)
add_dependencies(mace_static combined)

set_target_properties(mace_static
        PROPERTIES
        IMPORTED_LOCATION ${STATIC_LIB}
        )

install(TARGETS mace LIBRARY DESTINATION lib)
install(FILES ${STATIC_LIB} DESTINATION lib)
