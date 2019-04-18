include(ExternalProject)

set(OPENCL_CLHPP_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/opencl-clhpp")
set(OPENCL_CLHPP_INSTALL_DIR "${MACE_THIRD_PARTY_DIR}/install/opencl-clhpp")
set(OPENCL_CLHPP_INCLUDE_DIR "${OPENCL_CLHPP_INSTALL_DIR}" CACHE PATH "opencl-clhpp include directory." FORCE)

include_directories(SYSTEM ${OPENCL_CLHPP_INCLUDE_DIR})

# Mirror of https://github.com/KhronosGroup/OpenCL-CLHPP/archive/4c6f7d56271727e37fb19a9b47649dd175df2b12.zip
set(OPENCL_CLHPP_URL  "https://cnbj1.fds.api.xiaomi.com/mace/third-party/OpenCL-CLHPP/OpenCL-CLHPP-4c6f7d56271727e37fb19a9b47649dd175df2b12.zip")
set(OPENCL_CLHPP_HASH "SHA256=dab6f1834ec6e3843438cc0f97d63817902aadd04566418c1fcc7fb78987d4e7")

ExternalProject_Add(
  opencl_clhpp
  URL_HASH          "${OPENCL_CLHPP_HASH}"
  URL               "${OPENCL_CLHPP_URL}"
  PREFIX            "${OPENCL_CLHPP_SRCS_DIR}"
  CMAKE_ARGS        -DBUILD_DOCS=OFF
                    -DBUILD_EXAMPLES=OFF
                    -DBUILD_TESTS=OFF
                    -DCMAKE_INSTALL_PREFIX=${OPENCL_CLHPP_INSTALL_DIR}
                    ${THIRD_PARTY_EXTRA_CMAKE_ARGS}
)

add_dependencies(opencl_clhpp opencl_headers)
