cmake_minimum_required(VERSION 3.13)
project(METIS)

# FIXME: update the METIS cmake

set(GKLIB_PATH "GKlib" CACHE PATH "path to GKlib")
set(SHARED FALSE CACHE BOOL "build a shared library")
set(CMAKE_MACOSX_RPATH TRUE)

set ( CMAKE_INSTALL_RPATH ${CMAKE_SOURCE_DIR}/../lib )
set ( CMAKE_BUILD_RPATH ${CMAKE_SOURCE_DIR}/../lib )
set ( CMAKE_BUILD_WITH_INSTALL_RPATH TRUE )

if(MSVC)
  set(METIS_INSTALL FALSE)
else()
  set(METIS_INSTALL TRUE)
endif()

# Configure libmetis library.
if(SHARED)
  set(METIS_LIBRARY_TYPE SHARED)
else()
  set(METIS_LIBRARY_TYPE STATIC)
endif(SHARED)

include(${GKLIB_PATH}/GKlibSystem.cmake)
# Add include directories.
include_directories(${GKLIB_PATH})
include_directories(include)
# Recursively look for CMakeLists.txt in subdirs.
add_subdirectory("include")
add_subdirectory("libmetis")
add_subdirectory("programs")
