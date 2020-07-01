add_subdirectory(ccutils)

if(NOT HEXAGON)
  include(${PROJECT_SOURCE_DIR}/third_party/googletest/googletest.cmake)
  add_subdirectory(ccunit)
endif()

add_subdirectory(ccbenchmark)
add_subdirectory(ccbaseline)
