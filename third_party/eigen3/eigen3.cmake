INCLUDE(ExternalProject)

set(EIGEN3_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/eigen3")
set(EIGEN3_INCLUDE_DIR "${EIGEN3_SRCS_DIR}/src/eigen3" CACHE PATH "eigen3 include directory." FORCE)

include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# Mirror of https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz
set(EIGEN3_URL  "http://cnbj1.fds.api.xiaomi.com/mace/third-party/eigen/f3a22f35b044.tar.gz")
set(EIGEN3_HASH "SHA256=ca7beac153d4059c02c8fc59816c82d54ea47fe58365e8aded4082ded0b820c4")

ExternalProject_Add(
  eigen3
  URL_HASH          "${EIGEN3_HASH}"
  URL               "${EIGEN3_URL}"
  PREFIX            "${EIGEN3_SRCS_DIR}"
  BINARY_DIR        ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
