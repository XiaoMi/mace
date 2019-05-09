include_directories("${CMAKE_CURRENT_SOURCE_DIR}/ccutils")

add_subdirectory(ccutils)

if(MACE_ENABLE_TESTS)
  add_subdirectory(ccunit)
endif()

if(MACE_ENABLE_BENCHMARKS)
  add_subdirectory(ccbenchmark)
endif()
