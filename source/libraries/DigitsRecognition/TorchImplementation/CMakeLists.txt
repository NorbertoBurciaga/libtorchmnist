# Declare the library
set ( torch_implementation_sources
	TorchImplementation.cpp
)

add_library(TorchImplementation STATIC ${torch_implementation_sources})
target_link_libraries(TorchImplementation ${TORCH_LIBRARIES})
install(TARGETS TorchImplementation DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY . DESTINATION includes ${CMAKE_INSTALL_INCLUDEDIR} FILES_MATCHING PATTERN "*.h")