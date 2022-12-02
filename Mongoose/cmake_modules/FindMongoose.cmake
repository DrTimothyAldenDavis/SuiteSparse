#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/FindMongoose.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

# Finds the Mongoose include file and compiled library and sets:

# MONGOOSE_INCLUDE_DIR - where to find Mongoose.hpp
# MONGOOSE_LIBRARY     - compiled Mongoose library
# MONGOOSE_LIBRARIES   - libraries when using Mongoose
# MONGOOSE_FOUND       - true if Mongoose found

# set ``MONGOOSE_ROOT`` to a MONGOOSE installation root to
# tell this module where to look.

# To use this file in your application, copy this file into MyApp/cmake_modules
# where MyApp is your application and add the following to your
# MyApp/CMakeLists.txt file:
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake_modules")
#
# or, assuming MyApp and SuiteSparse sit side-by-side in a common folder, you
# can leave this file in place and use this command (revise as needed):
#
#   set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
#       "${CMAKE_SOURCE_DIR}/../SuiteSparse/SuiteSparse_config/cmake_modules")

#-------------------------------------------------------------------------------

# include files for Mongoose
find_path ( MONGOOSE_INCLUDE_DIR
    NAMES Mongoose.hpp
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/Mongoose
    HINTS ${CMAKE_SOURCE_DIR}/../Mongoose
    PATHS MONGOOSE_ROOT ENV MONGOOSE_ROOT
    PATH_SUFFIXES include Include
)

# compiled libraries Mongoose
find_library ( MONGOOSE_LIBRARY
    NAMES mongoose
    HINTS ${CMAKE_SOURCE_DIR}/..
    HINTS ${CMAKE_SOURCE_DIR}/../SuiteSparse/Mongoose
    HINTS ${CMAKE_SOURCE_DIR}/../Mongoose
    PATHS MONGOOSE_ROOT ENV MONGOOSE_ROOT
    PATH_SUFFIXES lib build alternative
)

# get version of the library
foreach(_VERSION VERSION_MAJOR VERSION_MINOR VERSION_PATCH)
  file (STRINGS ${MONGOOSE_INCLUDE_DIR}/Mongoose_Version.hpp _VERSION_LINE REGEX "define[ ]+Mongoose_${_VERSION}")
  if(_VERSION_LINE)
    string (REGEX REPLACE ".*define[ ]+Mongoose_${_VERSION}[ ]+([0-9]*).*" "\\1" _MONGOOSE_${_VERSION} "${_VERSION_LINE}")
  endif()
  unset(_VERSION_LINE)
endforeach()
set (MONGOOSE_VERSION "${_MONGOOSE_VERSION_MAJOR}.${_MONGOOSE_VERSION_MINOR}.${_MONGOOSE_VERSION_PATCH}")
set (MONGOOSE_LIBRARIES ${MONGOOSE_LIBRARY})

include (FindPackageHandleStandardArgs)

find_package_handle_standard_args ( Mongoose
    REQUIRED_VARS MONGOOSE_LIBRARIES MONGOOSE_INCLUDE_DIR
    VERSION_VAR MONGOOSE_VERSION
)

mark_as_advanced (
    MONGOOSE_INCLUDE_DIR
    MONGOOSE_LIBRARY
    MONGOOSE_LIBRARIES
)

if ( MONGOOSE_FOUND )
    message ( STATUS "Mongoose include dir: ${MONGOOSE_INCLUDE_DIR}" )
    message ( STATUS "Mongoose library:     ${MONGOOSE_LIBRARY}" )
    message ( STATUS "Mongoose version:     ${MONGOOSE_VERSION}" )
else ( )
    message ( STATUS "Mongoose not found" )
endif ( )

