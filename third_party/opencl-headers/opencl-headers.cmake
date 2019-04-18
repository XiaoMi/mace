include(ExternalProject)

set(OPENCL_HEADERS_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/opencl-headers")
set(OPENCL_HEADERS_INCLUDE_DIR "${OPENCL_HEADERS_SRCS_DIR}/src/opencl_headers/opencl20" CACHE PATH "opencl-headers include directory." FORCE)

include_directories(SYSTEM ${OPENCL_HEADERS_INCLUDE_DIR})

# Mirror of https://github.com/KhronosGroup/OpenCL-Headers/archive/f039db6764d52388658ef15c30b2237bbda49803.zip
set(OPENCL_HEADERS_URL  "https://cnbj1.fds.api.xiaomi.com/mace/third-party/OpenCL-Headers/f039db6764d52388658ef15c30b2237bbda49803.zip")
set(OPENCL_HEADERS_HASH "SHA256=b2b813dd88a7c39eb396afc153070f8f262504a7f956505b2049e223cfc2229b")

ExternalProject_Add(
  opencl_headers
  URL_HASH          "${OPENCL_HEADERS_HASH}"
  URL               "${OPENCL_HEADERS_URL}"
  PREFIX            "${OPENCL_HEADERS_SRCS_DIR}"
  BINARY_DIR        ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
