include(ExternalProject)

set(TFLITE_SRCS_DIR    "${MACE_THIRD_PARTY_DIR}/tflite")
set(TFLITE_INCLUDE_DIR "${TFLITE_SRCS_DIR}/src/tflite" CACHE PATH "tflite include directory." FORCE)

include_directories(SYSTEM ${TFLITE_INCLUDE_DIR})

set(TFLITE_URL  "http://cnbj1.fds.api.xiaomi.com/mace/third-party/tflite/tensorflow-mace-d73e88fc830320d3818ac24e57cd441820a85cc9.zip")
set(TFLITE_HASH "SHA256=6f2671a02fe635a82c289c8c40a6e5bc24670ff1d4c3c2ab4a7aa9b825256a18")

ExternalProject_Add(
  tflite
  URL_HASH          "${TFLITE_HASH}"
  URL               "${TFLITE_URL}"
  PREFIX            "${TFLITE_SRCS_DIR}"
  BINARY_DIR        ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)
