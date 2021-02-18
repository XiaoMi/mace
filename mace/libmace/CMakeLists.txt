set(LIBMACE_SRCS capability.cc mace.cc)

add_library(mace SHARED ${LIBMACE_SRCS})
target_link_libraries(mace ops)
if(NOT APPLE)
  set_target_properties(mace PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mace_version_script.lds")
endif(NOT APPLE)

add_library(mace_API STATIC ${LIBMACE_SRCS})
target_link_libraries(mace_API ops)

set(STATIC_LIB ${CMAKE_BINARY_DIR}/mace/libmace/libmace_static.a)
set(BASE_PATH ${CMAKE_BINARY_DIR})
set(PORT_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/port/posix/libport_posix.a\n"
              "addlib ${CMAKE_BINARY_DIR}/mace/port/linux_base/libport_linux_base.a\n")
if(ANDROID)
  set(PORT_LIBS ${PORT_LIBS}
                "addlib ${CMAKE_BINARY_DIR}/mace/port/android/libport_android.a")
  if(MACE_ENABLE_RPCMEM)
    set(PORT_LIBS ${PORT_LIBS}
                  "\naddlib ${PROJECT_SOURCE_DIR}/third_party/rpcmem/${ANDROID_ABI}/rpcmem.a")
  endif()
  
elseif(WIN32)
  set(STATIC_LIB ${CMAKE_BINARY_DIR}/mace/libmace/libmace_static.lib)
else(WIN32)
  set(PORT_LIBS ${PORT_LIBS}
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux/libport_linux.a")
endif(ANDROID)

string(CONCAT PORT_LIBS_MRI ${PORT_LIBS} "")
configure_file("${CMAKE_SOURCE_DIR}/mace/libmace/mace.mri.in" 
               "${CMAKE_BINARY_DIR}/mace/libmace/mace.mri")

if(APPLE)
  add_custom_target(combined ALL
  COMMAND ${_CMAKE_TOOLCHAIN_PREFIX}libtool -static -o ${STATIC_LIB} 
          $<TARGET_FILE:mace_API> 
          $<TARGET_FILE:ops> 
          $<TARGET_FILE:core> 
          $<TARGET_FILE:utils> 
          $<TARGET_FILE:proto> 
          $<TARGET_FILE:generated_version> 
          $<TARGET_FILE:generated_opencl_kernel> 
          $<TARGET_FILE:libprotobuf_lite> 
          $<TARGET_FILE:port_base> 
          $<TARGET_FILE:port_posix> 
          $<TARGET_FILE:port_darwin>
  DEPENDS mace_API ops core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_darwin
  )
elseif(WIN32)
  add_custom_target(combined ALL
  COMMAND ${_CMAKE_TOOLCHAIN_PREFIX}LIB.EXE /OUT:${STATIC_LIB} 
          $<TARGET_FILE:mace_API> 
          $<TARGET_FILE:ops> 
          $<TARGET_FILE:core> 
          $<TARGET_FILE:utils> 
          $<TARGET_FILE:proto> 
          $<TARGET_FILE:generated_version> 
          $<TARGET_FILE:generated_opencl_kernel> 
          $<TARGET_FILE:libprotobuf_lite> 
          $<TARGET_FILE:port_base> 
          $<TARGET_FILE:port_windows> 
  DEPENDS mace_API ops core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_windows
  )
elseif(ANDROID)
  add_custom_target(combined ALL
    COMMAND ${_CMAKE_TOOLCHAIN_PREFIX}ar -M < ${CMAKE_BINARY_DIR}/mace/libmace/mace.mri
    DEPENDS mace_API ops core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_linux_base port_android
    )
else(ANDROID)
  add_custom_target(combined ALL
    COMMAND ar -M < ${CMAKE_BINARY_DIR}/mace/libmace/mace.mri
    DEPENDS mace_API ops core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_linux_base port_linux
    )
endif(APPLE)

if(MACE_ENABLE_RPCMEM)
  add_dependencies(combined rpcmem)
endif()

add_library(mace_static STATIC IMPORTED GLOBAL)
add_dependencies(mace_static combined)

set_target_properties(mace_static
        PROPERTIES
        IMPORTED_LOCATION ${STATIC_LIB}
        )

install(TARGETS mace LIBRARY DESTINATION lib)
install(FILES ${STATIC_LIB} DESTINATION lib)
