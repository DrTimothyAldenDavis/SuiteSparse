#-------------------------------------------------------------------------------
# SuiteSparse/cmake_modules/SuiteSparsePolicy.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# cmake policies for all of SuiteSparse
cmake_minimum_required ( VERSION 3.19 )

message ( STATUS "Source:        ${CMAKE_SOURCE_DIR} ")
message ( STATUS "Build:         ${CMAKE_BINARY_DIR} ")
# message ( STATUS "Modules:       ${CMAKE_MODULE_PATH}")

cmake_policy ( SET CMP0042 NEW )    # enable MACOSX_RPATH by default
cmake_policy ( SET CMP0048 NEW )    # VERSION variable policy
cmake_policy ( SET CMP0054 NEW )    # if ( expression ) handling policy

set ( CMAKE_MACOSX_RPATH TRUE )
enable_language ( C )
include ( GNUInstallDirs )

# and the ./build folder to the runpath so other SuiteSparse packages can
# find this one with out "make install"
set ( CMAKE_BUILD_RPATH ${CMAKE_BUILD_RPATH} ${CMAKE_BINARY_DIR} )

# determine if this package is inside the top-level SuiteSparse folder
# (if ../lib and ../include exist, relative to the source directory)
set ( INSIDE_SUITESPARSE 
        ( ( EXISTS ${CMAKE_SOURCE_DIR}/../lib     ) AND 
        (   EXISTS ${CMAKE_SOURCE_DIR}/../include ) ) )

if ( INSIDE_SUITESPARSE )
    # ../lib and ../include exist: the package is inside SuiteSparse.
    # find ( REAL_PATH ...) requires cmake 3.19.
    file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../lib     LOCAL_LIBDIR )
    file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../include LOCAL_INCLUDEDIR )
    file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../bin     LOCAL_BINDIR )
    message ( STATUS "Local install: ${LOCAL_LIBDIR} ")
    message ( STATUS "Local include: ${LOCAL_INCLUDEDIR} ")
    message ( STATUS "Local bin:     ${LOCAL_BINDIR} ")
    # append ../lib to the install and build runpaths 
    set ( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${LOCAL_LIBDIR} )
    set ( CMAKE_BUILD_RPATH   ${CMAKE_BUILD_RPATH}   ${LOCAL_LIBDIR} )
endif ( )

message ( STATUS "Install rpath: ${CMAKE_INSTALL_RPATH} ")
message ( STATUS "Build   rpath: ${CMAKE_BUILD_RPATH} ")

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

message ( STATUS "Build type:    ${CMAKE_BUILD_TYPE} ")

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

#-------------------------------------------------------------------------------
# BLAS library
#-------------------------------------------------------------------------------

# Uncomment one of the lines below to select OpenBLAS or the Intel MKL.
# Be sure to use a parallel BLAS library for best results in UMFPACK,
# CHOLMOD, SPQR, and ParU.  All SuiteSparse packages should use the same
# BLAS and the same OpenMP library.

# set ( BLA_VENDOR OpenBLAS )
# set ( BLA_VENDOR Intel10_64ilp )

# The Intel MKL BLAS is recommended.  It is free to download (but be sure to
# check their license to make sure you accept it).   See:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.htm

