add_library(apu_flow STATIC
  apu_ref_flow.cc
)

target_link_libraries(apu_flow core ${EXTRA_LINK_LIBS})

install(TARGETS apu_flow ARCHIVE DESTINATION lib)
