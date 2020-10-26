add_subdirectory(ccutils)

if(NOT HEXAGON)
  add_subdirectory(ccunit)
endif()

add_subdirectory(ccbenchmark)
add_subdirectory(ccbaseline)
