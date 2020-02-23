include_directories(${CMAKE_SOURCE_DIR}/source/libraries/DigitsRecognition/TorchImplementation)

add_executable(${APPLICATION_NAME} main)
target_link_libraries(${APPLICATION_NAME} TorchImplementation)
install(TARGETS ${APPLICATION_NAME} DESTINATION ${CMAKE_INSTALL_BINDIR})
