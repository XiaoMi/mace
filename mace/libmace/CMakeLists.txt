set(LIBMACE_SRCS
  capability.cc
  gpu_context_builder.cc
  mace_engine.cc
  mace_engine_config.cc
  mace_tensor.cc
  engines/base_engine.cc
  engines/engine_registry.cc
  engines/serial_engine.cc
  engines/single_flow_engine.cc
)

add_library(mace SHARED ${LIBMACE_SRCS})
target_link_libraries(mace ops flows runtimes generated_version)

if(MACE_ENABLE_RPCMEM)
  target_link_libraries(mace rpcmems)
endif()

if(NOT APPLE)
  set_target_properties(mace PROPERTIES LINK_FLAGS
    "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/mace_version_script.lds")
endif(NOT APPLE)

add_library(mace_API STATIC ${LIBMACE_SRCS})
target_link_libraries(mace_API ops flows runtimes)

set(STATIC_LIB ${CMAKE_BINARY_DIR}/mace/libmace/libmace_static.a)
set(BASE_PATH ${CMAKE_BINARY_DIR})

set(PORT_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/port/posix/libport_posix.a\n")
if(ANDROID)
  set(PORT_LIBS ${PORT_LIBS}
                "addlib ${CMAKE_BINARY_DIR}/mace/port/android/libport_android.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux_base/libport_linux_base.a\n")
  if(MACE_ENABLE_RPCMEM)
    set(PORT_LIBS ${PORT_LIBS}
                  "\naddlib ${PROJECT_SOURCE_DIR}/third_party/rpcmem/${ANDROID_ABI}/rpcmem.a")
  endif()
elseif(QNX)
  set(PORT_LIBS ${PORT_LIBS} "addlib ${CMAKE_BINARY_DIR}/mace/port/qnx/libport_qnx.a")
elseif(WIN32)
  set(STATIC_LIB ${CMAKE_BINARY_DIR}/mace/libmace/libmace_static.lib)
else(WIN32)
  set(PORT_LIBS ${PORT_LIBS}
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux/libport_linux.a\n"
                "addlib ${CMAKE_BINARY_DIR}/mace/port/linux_base/libport_linux_base.a\n")
endif(ANDROID)

set(FLOW_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/flows/libflows.a\n"
              "addlib ${CMAKE_BINARY_DIR}/mace/flows/cpu/libcpu_flow.a\n")
set(RUNTIME_LIBS "addlib ${CMAKE_BINARY_DIR}/mace/runtimes/libruntimes.a\n"
                 "addlib ${CMAKE_BINARY_DIR}/mace/runtimes/cpu/libcpu_runtime.a\n")

if(MACE_ENABLE_OPENCL)
  set(FLOW_LIBS ${FLOW_LIBS}
                "addlib ${CMAKE_BINARY_DIR}/mace/flows/opencl/libopencl_flow.a\n")
  set(RUNTIME_LIBS ${RUNTIME_LIBS}
                "addlib ${CMAKE_BINARY_DIR}/mace/runtimes/opencl/libopencl_runtime.a\n")
endif(MACE_ENABLE_OPENCL)

if(MACE_ENABLE_HEXAGON)
  set(FLOW_LIBS ${FLOW_LIBS}
          "addlib ${CMAKE_BINARY_DIR}/mace/flows/hexagon/libhexagon_flow.a\n")
  set(RUNTIME_LIBS ${RUNTIME_LIBS}
          "addlib ${CMAKE_BINARY_DIR}/mace/runtimes/hexagon/libhexagon_runtime.a\n")
endif(MACE_ENABLE_HEXAGON)

if(MACE_ENABLE_MTK_APU)
  set(FLOW_LIBS ${FLOW_LIBS}
          "addlib ${CMAKE_BINARY_DIR}/mace/flows/apu/libapu_flow.a\n")
  set(RUNTIME_LIBS ${RUNTIME_LIBS}
          "addlib ${CMAKE_BINARY_DIR}/mace/runtimes/apu/libapu_runtime.a\n")
endif(MACE_ENABLE_MTK_APU)

string(CONCAT PORT_LIBS_MRI ${PORT_LIBS} "")
string(CONCAT FLOW_LIBS_MRI ${FLOW_LIBS} "")
string(CONCAT RUNTIME_LIBS_MRI ${RUNTIME_LIBS} "")
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
  DEPENDS mace_API ops flows runtimes core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_darwin
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
  DEPENDS mace_API ops flows runtimes core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_windows
  )
elseif(ANDROID)
  add_custom_target(combined ALL
    COMMAND ${_CMAKE_TOOLCHAIN_PREFIX}ar -M < ${CMAKE_BINARY_DIR}/mace/libmace/mace.mri
    DEPENDS mace_API ops flows runtimes core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_linux_base port_android
    )
elseif(QNX)
  add_custom_target(combined ALL
    COMMAND ${CROSSTOOL_ROOT}/qnx700/host/linux/x86_64/usr/bin/ntoaarch64-ar -M < ${CMAKE_BINARY_DIR}/mace/libmace/mace.mri
    DEPENDS mace_API ops flows runtimes core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_qnx
    )
else() #linux
  add_custom_target(combined ALL
    COMMAND ar -M < ${CMAKE_BINARY_DIR}/mace/libmace/mace.mri
    DEPENDS mace_API ops flows runtimes core utils proto generated_version generated_opencl_kernel libprotobuf_lite port_base port_posix port_linux_base port_linux
    )
endif(APPLE)

if(MACE_ENABLE_RPCMEM)
  add_dependencies(combined rpcmem)
endif()

add_library(mace_static STATIC IMPORTED GLOBAL)
add_dependencies(mace_static combined)
set_target_properties(mace_static PROPERTIES
  IMPORTED_LOCATION ${STATIC_LIB}
)

if(ANDROID)
  set_target_properties(mace_static PROPERTIES
    INTERFACE_LINK_LIBRARIES log
  )
endif()

set_target_properties(mace_static
        PROPERTIES
        IMPORTED_LOCATION ${STATIC_LIB}
        )

install(TARGETS mace LIBRARY DESTINATION lib)
install(FILES ${STATIC_LIB} DESTINATION lib)

if(ANDROID_STL)
  if (${ANDROID_STL} STREQUAL "c++_shared") #
    install(FILES "${ANDROID_NDK}/sources/cxx-stl/llvm-libc++/libs/${ANDROID_ABI}/libc++_shared.so" DESTINATION lib)
  elseif(${ANDROID_STL} STREQUAL "stlport_shared") #
    install(FILES "${ANDROID_NDK}/sources/cxx-stl/stlport/libs/${ANDROID_ABI}/libstlport_shared.so" DESTINATION lib)
  elseif(${ANDROID_STL} STREQUAL "gnustl_shared") #
    install(FILES "${ANDROID_NDK}/sources/cxx-stl/gnu-libstdc++/4.9/libs/${ANDROID_ABI}/libgnustl_shared.so" DESTINATION lib)
  endif()
endif()
