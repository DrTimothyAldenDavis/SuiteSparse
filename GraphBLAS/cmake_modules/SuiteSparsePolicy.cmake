#-------------------------------------------------------------------------------
# SuiteSparse/cmake_modules/SuiteSparsePolicy.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# cmake policies for all of SuiteSparse
# cmake 3.19 is required

message ( STATUS "Source:        " ${CMAKE_SOURCE_DIR} )
message ( STATUS "Build:         " ${CMAKE_BINARY_DIR} )

cmake_policy ( SET CMP0042 NEW )    # enable MACOSX_RPATH by default
cmake_policy ( SET CMP0048 NEW )    # VERSION variable policy
cmake_policy ( SET CMP0054 NEW )    # if ( expression ) handling policy

set ( CMAKE_MACOSX_RPATH TRUE )
enable_language ( C )
include ( GNUInstallDirs )

# determine if this package is inside the top-level SuiteSparse folder
# (if ../lib and ../include exist, relative to the source directory)
set ( INSIDE_SUITESPARSE 
        ( ( EXISTS ${CMAKE_SOURCE_DIR}/../lib     ) AND 
        (   EXISTS ${CMAKE_SOURCE_DIR}/../include ) ) )

if ( INSIDE_SUITESPARSE )
    # ../lib and ../include exists
    file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../lib     LOCAL_LIBDIR )
    file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../include LOCAL_INCLUDEDIR )
    message ( STATUS "Local install: " ${LOCAL_LIBDIR} )
    message ( STATUS "Local include: " ${LOCAL_INCLUDEDIR} )
    set ( CMAKE_INSTALL_RPATH ${LOCAL_LIBDIR} )
    set ( CMAKE_BUILD_RPATH   ${LOCAL_LIBDIR} )
    set ( CMAKE_BUILD_WITH_INSTALL_RPATH TRUE )
endif ( )

message ( STATUS "Install rpath: " ${CMAKE_INSTALL_RPATH} )
message ( STATUS "Build   rpath: " ${CMAKE_BUILD_RPATH} )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

message ( STATUS "Build type:    " ${CMAKE_BUILD_TYPE} )

