cmake_minimum_required (VERSION 3.5)

#---------------------------------------- Project ----------------------------------------
project (libtorchmnist)

# Version
set (VERSION_MAJOR 0)
set (VERSION_MINOR 0)
set (VERSION_PATCH 1)

# Disable in-source builds to prevent source tree corruption.
if(" ${CMAKE_SOURCE_DIR}" STREQUAL " ${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "
FATAL: In-source builds are not allowed.
       You should create a separate directory for build files.
")
endif()

#---------------------------------------- Options ----------------------------------------
option(DOWNLOAD_MNIST "Download the MNIST dataset from the internet" ON)

#---------------------------------------- Settings ---------------------------------------
# Output directory settings
set(output_dir ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${output_dir})

# -std=c++1z or -std=gnu++1z
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#---------------------------------------- Packages ---------------------------------------
# Find torch
find_package(Torch REQUIRED)

#---------------------------------------- Download data ----------------------------------
if (DOWNLOAD_MNIST)
  message(STATUS "Downloading MNIST dataset")
  execute_process(
    COMMAND python ${CMAKE_CURRENT_LIST_DIR}/download_mnist.py
      -d ${CMAKE_BINARY_DIR}/data
    ERROR_VARIABLE DOWNLOAD_ERROR)
  if (DOWNLOAD_ERROR)
    message(FATAL_ERROR "Error downloading MNIST dataset: ${DOWNLOAD_ERROR}")
  endif()
endif()

#---------------------------------------- Directories ------------------------------------
# add the CMakeLists.txt files located in subdirectories
add_subdirectory(source)
