include (ExternalProject)

set(PROTOBUF_SRCS_DIR     "${MACE_THIRD_PARTY_DIR}/protobuf")
set(PROTOBUF_INSTALL_DIR  "${MACE_THIRD_PARTY_DIR}/install/protobuf")
set(PROTOBUF_INCLUDE_DIR  "${PROTOBUF_INSTALL_DIR}/include")

set(PROTOC_SRCS_DIR       "${MACE_THIRD_PARTY_DIR}/protoc")
set(PROTOC_INSTALL_DIR    "${MACE_THIRD_PARTY_DIR}/install/protoc")

include_directories(SYSTEM ${PROTOBUF_INCLUDE_DIR})

if(MSVC)
  set(PROTOBUF_LITE_LIBRARIES
    "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.lib" CACHE FILEPATH "libprotobuf lite libraries." FORCE)
  set(PROTOC_BIN
    "${PROTOC_INSTALL_DIR}/bin/protoc.exe" CACHE FILEPATH "protoc compiler." FORCE)
else(MSVC)
  set(PROTOBUF_LITE_LIBRARIES
    "${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.a" CACHE FILEPATH "libprotobuf lite libraries." FORCE)
  set(PROTOC_BIN
    "${PROTOC_INSTALL_DIR}/bin/protoc" CACHE FILEPATH "protoc compiler." FORCE)
endif(MSVC)

# Mirror of https://github.com/google/protobuf/archive/v3.6.1.zip
set(PROTOBUF_URL    "https://cnbj1.fds.api.xiaomi.com/mace/third-party/protobuf/protobuf-3.6.1.zip")
set(PROTOBUF_HASH   "SHA256=d7a221b3d4fb4f05b7473795ccea9e05dab3b8721f6286a95fffbffc2d926f8b")

ExternalProject_Add(
  protobuf
  URL_HASH          "${PROTOBUF_HASH}"
  URL               "${PROTOBUF_URL}"
  PREFIX            "${PROTOBUF_SRCS_DIR}"
  BUILD_BYPRODUCTS  ${PROTOBUF_LITE_LIBRARIES}
  SOURCE_DIR        "${PROTOBUF_SRCS_DIR}/src/protobuf"
  CONFIGURE_COMMAND ${CMAKE_COMMAND} ../protobuf/cmake/
  #-DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_INSTALL_PREFIX=${PROTOBUF_INSTALL_DIR}
                    -DCMAKE_VERBOSE_MAKEFILE=OFF
                    -Dprotobuf_BUILD_TESTS=OFF
                    -Dprotobuf_WITH_ZLIB=OFF
                    -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
                    -Dprotobuf_MSVC_STATIC_RUNTIME=OFF
                    -DCMAKE_GENERATOR=${CMAKE_GENERATOR}
                    ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
)

if(MSVC)
  add_custom_command(TARGET protobuf POST_BUILD
    COMMAND if $<CONFIG:Debug>==1 (cmake -E copy ${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lited.lib ${PROTOBUF_INSTALL_DIR}/lib/libprotobuf-lite.lib)
  )
endif(MSVC)

add_library(libprotobuf_lite STATIC IMPORTED GLOBAL)
set_property(TARGET libprotobuf_lite PROPERTY IMPORTED_LOCATION ${PROTOBUF_LITE_LIBRARIES})
add_dependencies(libprotobuf_lite protobuf)

install(FILES ${PROTOBUF_LITE_LIBRARIES} DESTINATION lib)

set(BUILD_PROTOC TRUE)
if(COMMAND protoc)
  execute_process(COMMAND protoc OUTPUT_VARIABLE PROTOC_VER)
  if(${PROTOC_VER} STREQUAL "libprotoc 3.6.1")
    set(PROTOC_BIN protoc CACHE FILEPATH "protoc compiler." FORCE)
    set(BUILD_PROTOC FALSE)
    add_custom_target(protoc_bin COMMENT "protoc noop target")
  endif(${PROTOC_VER} STREQUAL "libprotoc 3.6.1")
endif(COMMAND protoc)

if(BUILD_PROTOC)
  if(NOT APPLE)
    # Actually this works for iOS build on macOS, but the compatibility is not
    # thoroughly tested, so we use the downloaded version instead.
    set(PROTOC_CMAKE_GENERATOR ${CMAKE_GENERATOR})
    if(APPLE AND ${CMAKE_GENERATOR} STREQUAL "Xcode")
      # Force Xcode to build protoc on host
      set(PROTOC_CMAKE_GENERATOR "Unix Makefiles")
      execute_process(COMMAND sw_vers -productVersion OUTPUT_VARIABLE MACOS_PRODVER)
      set(EXTRA_MACOS_CMAKE_ARGS
        -DCMAKE_OSX_SYSROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk
        -DCMAKE_OSX_DEPLOYMENT_TARGET=${MACOS_PRODVER}
      )
    endif(APPLE AND ${CMAKE_GENERATOR} STREQUAL "Xcode")

    ExternalProject_Add(
      protoc_bin
      URL_HASH          "${PROTOBUF_HASH}"
      URL               "${PROTOBUF_URL}"
      PREFIX            "${PROTOC_SRCS_DIR}"
      BUILD_BYPRODUCTS  "${PROTOC_BIN}"
      SOURCE_DIR        "${PROTOC_SRCS_DIR}/src/protoc"
      CONFIGURE_COMMAND ${CMAKE_COMMAND} ../protoc/cmake/
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_INSTALL_PREFIX=${PROTOC_INSTALL_DIR}
      -DCMAKE_VERBOSE_MAKEFILE=OFF
      -Dprotobuf_BUILD_TESTS=OFF
      -Dprotobuf_WITH_ZLIB=OFF
      -Dprotobuf_BUILD_PROTOC_BINARIES=ON
      -DCMAKE_GENERATOR=${PROTOC_CMAKE_GENERATOR}
      ${EXTRA_MACOS_CMAKE_ARGS}
    )
  else(APPLE)
    # This is backup protoc when build doesn't work for macOS+iOS
    # Mirror of "https://github.com/protocolbuffers/protobuf/releases/download/v3.6.1/protoc-3.6.1-osx-x86_64.zip"
    set(PROTOC_URL  "https://cnbj1.fds.api.xiaomi.com/mace/third-party/protobuf/protoc-3.6.1-osx-x86_64.zip")
    set(PROTOC_HASH "SHA256=0decc6ce5beed07f8c20361ddeb5ac7666f09cf34572cca530e16814093f9c0c")
    ExternalProject_Add(
      protoc_bin
      URL_HASH          "${PROTOC_HASH}"
      URL               "${PROTOC_URL}"
      PREFIX            "${PROTOC_SRCS_DIR}"
      BINARY_DIR        ""
      CONFIGURE_COMMAND ""
      BUILD_COMMAND     ""
      INSTALL_COMMAND   ""
      TEST_COMMAND      ""
    )
    set(PROTOC_BIN
      "${PROTOC_SRCS_DIR}/src/protoc_bin/bin/protoc" CACHE FILEPATH "protoc compiler." FORCE)
  endif(NOT APPLE)
endif(BUILD_PROTOC)
