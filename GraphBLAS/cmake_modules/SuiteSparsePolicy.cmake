#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparsePolicy.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2022, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# SuiteSparse CMake policies.  The following parameters can be defined prior
# to including this file:
#
#   CMAKE_BUILD_TYPE:   if not set, it is set below to "Release".
#                       To use the "Debug" policy, precede this with
#                       set ( CMAKE_BUILD_TYPE Debug )
#
#   ENABLE_CUDA:        if set to true, CUDA is enabled for the project.
#                       Default: true for CHOLMOD and SPQR
#
#   GLOBAL_INSTALL:     if true, "make install" will
#                       into /usr/local/lib and /usr/local/include.
#                       Default: true
#
#   LOCAL_INSTALL:      if true, "make install" will
#                       into SuiteSparse/lib and SuiteSparse/include,
#                       but these folders must also already exist.
#                       Default: false
#
#   NSTATIC:            if true, static libraries are not built.
#                       Default: false, except for GraphBLAS, which
#                       takes a long time to compile so the default for
#                       GraphBLAS is true.  For Mongoose, the NSTATIC setting
#                       is treated as if it always false, since the mongoose
#                       program is built with the static library.
#
#   NPARTITION:         if true, SuiteSparse_metis will not be compiled or used.
#                       Default: false
#
#   SUITESPARSE_CUDA_ARCHITECTURES:  a string, such as "all" or
#                       "35;50;75;80" that lists the CUDA architectures to use
#                       when compiling CUDA kernels with nvcc.  The "all"
#                       option requires cmake 3.23 or later.
#                       Default: "52;75;80".
#
# To select a specific BLAS library, edit the SuiteSparseBLAS.cmake file.

cmake_minimum_required ( VERSION 3.19 )

message ( STATUS "Source:        ${CMAKE_SOURCE_DIR} ")
message ( STATUS "Build:         ${CMAKE_BINARY_DIR} ")

cmake_policy ( SET CMP0042 NEW )    # enable MACOSX_RPATH by default
cmake_policy ( SET CMP0048 NEW )    # VERSION variable policy
cmake_policy ( SET CMP0054 NEW )    # if ( expression ) handling policy
cmake_policy ( SET CMP0104 NEW )    # initialize CUDA architectures

# look for cmake modules installed by prior compilations of SuiteSparse packages
set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules )

if ( NOT DEFINED NPARTITION )
    set ( NPARTITION false )
endif ( )
if ( NPARTITION )
    add_compile_definitions ( NPARTITION )
endif ( )

if ( SUITESPARSE_SECOND_LEVEL )
    # some packages in SuiteSparse is in SuiteSparse/Package/Package
    set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        ${CMAKE_SOURCE_DIR}/../../lib/cmake )
else ( )
    # most packages in SuiteSparse are located in SuiteSparse/Package
    set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
        ${CMAKE_SOURCE_DIR}/../lib/cmake )
endif ( )

set ( CMAKE_MACOSX_RPATH TRUE )
enable_language ( C )
include ( GNUInstallDirs )

# add the ./build folder to the runpath so other SuiteSparse packages can
# find this one without "make install"
set ( CMAKE_BUILD_RPATH ${CMAKE_BUILD_RPATH} ${CMAKE_BINARY_DIR} )

if ( NOT DEFINED NSTATIC )
    # default for all SuiteSparse packages (except GraphBLAS)
    # is to build the static libraries.
    set ( NSTATIC false )
endif ( )

# determine if this package is inside the top-level SuiteSparse folder
# (if ../lib and ../include exist, relative to the source directory)

if ( NOT DEFINED GLOBAL_INSTALL )
    # if not defined, GLOBAL_INSTALL is set to true.
    # "make install" will install in CMAKE_INSTALL_PREFIX
    set ( GLOBAL_INSTALL true )
endif ( )

if ( NOT DEFINED LOCAL_INSTALL )
    # if not defined, LOCAL_INSTALL is set to false
    # "make install" will install in SuiteSparse/[lib,include,bin],
    # if they exist.
    set ( LOCAL_INSTALL false )
endif ( )

set ( INSIDE_SUITESPARSE false )

if ( LOCAL_INSTALL )
    # if you do not want to install local copies of SuiteSparse
    # packages in SuiteSparse/lib and SuiteSparse/, set
    # LOCAL_INSTALL to false in your CMake options.
    if ( SUITESPARSE_SECOND_LEVEL )
        # the package is normally located at the 2nd level inside SuiteSparse
        # (SuiteSparse/GraphBLAS/GraphBLAS/ for example)
        if ( ( EXISTS ${CMAKE_SOURCE_DIR}/../../lib     ) AND
           (   EXISTS ${CMAKE_SOURCE_DIR}/../../include ) AND
           (   EXISTS ${CMAKE_SOURCE_DIR}/../../bin     ) )
            set ( INSIDE_SUITESPARSE true )
        endif ( )
    else ( )
        # typical case, the package is at the 1st level inside SuiteSparse
        # (SuiteSparse/AMD for example)
        if ( ( EXISTS ${CMAKE_SOURCE_DIR}/../lib     ) AND
             ( EXISTS ${CMAKE_SOURCE_DIR}/../include ) AND
             ( EXISTS ${CMAKE_SOURCE_DIR}/../bin     ) )
            set ( INSIDE_SUITESPARSE true )
        endif ( )
    endif ( )
endif ( )

if ( INSIDE_SUITESPARSE )
    # ../lib and ../include exist: the package is inside SuiteSparse.
    # find ( REAL_PATH ...) requires cmake 3.19.
    if ( SUITESPARSE_SECOND_LEVEL )
        file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../../lib     SUITESPARSE_LIBDIR )
        file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../../include SUITESPARSE_INCLUDEDIR )
        file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../../bin     SUITESPARSE_BINDIR )
    else ( )
        file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../lib     SUITESPARSE_LIBDIR )
        file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../include SUITESPARSE_INCLUDEDIR )
        file ( REAL_PATH  ${CMAKE_SOURCE_DIR}/../bin     SUITESPARSE_BINDIR )
    endif ( )
    message ( STATUS "Local install: ${SUITESPARSE_LIBDIR} ")
    message ( STATUS "Local include: ${SUITESPARSE_INCLUDEDIR} ")
    message ( STATUS "Local bin:     ${SUITESPARSE_BINDIR} ")
    # append ../lib to the install and build runpaths
    set ( CMAKE_INSTALL_RPATH ${CMAKE_INSTALL_RPATH} ${SUITESPARSE_LIBDIR} )
    set ( CMAKE_BUILD_RPATH   ${CMAKE_BUILD_RPATH}   ${SUITESPARSE_LIBDIR} )
endif ( )

message ( STATUS "Install rpath: ${CMAKE_INSTALL_RPATH} ")
message ( STATUS "Build   rpath: ${CMAKE_BUILD_RPATH} ")

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

message ( STATUS "Build type:    ${CMAKE_BUILD_TYPE} ")

set ( CMAKE_INCLUDE_CURRENT_DIR ON )

#-------------------------------------------------------------------------------
# find CUDA
#-------------------------------------------------------------------------------

if ( ENABLE_CUDA )

    # try finding CUDA
    include ( CheckLanguage )
    check_language ( CUDA )
    message ( STATUS "Looking for CUDA" )
    if ( CMAKE_CUDA_COMPILER )
        # with CUDA:
        message ( STATUS "Find CUDA tool kit:" )
        # FindCUDAToolKit needs to have C or CXX enabled first (see above)
        include ( FindCUDAToolkit )
        message ( STATUS "CUDA toolkit found:   " ${CUDAToolkit_FOUND} )
        message ( STATUS "CUDA toolkit version: " ${CUDAToolkit_VERSION} )
        message ( STATUS "CUDA toolkit include: " ${CUDAToolkit_INCLUDE_DIRS} )
        message ( STATUS "CUDA toolkit lib dir: " ${CUDAToolkit_LIBRARY_DIR} )
        if ( CUDAToolkit_VERSION VERSION_LESS "11.2" )
            # CUDA is present but too old
            message ( STATUS "CUDA: not enabled (CUDA 11.2 or later required)" )
            set ( SUITESPARSE_CUDA off )
        else ( )
            # CUDA 11.2 or later present
            enable_language ( CUDA )
            set ( SUITESPARSE_CUDA on )
        endif ( )
    else ( )
        # without CUDA:
        message ( STATUS "CUDA: not found" )
        set ( SUITESPARSE_CUDA off )
    endif ( )

else ( )

    # CUDA is disabled
    set ( SUITESPARSE_CUDA off )

endif ( )

if ( SUITESPARSE_CUDA )
    message ( STATUS "CUDA: enabled" )
    add_compile_definitions ( SUITESPARSE_CUDA )
    if ( NOT DEFINED SUITESPARSE_CUDA_ARCHITECTURES )
        # default architectures
        set ( CMAKE_CUDA_ARCHITECTURES "52;75;80" )
    else ( )
        set ( CMAKE_CUDA_ARCHITECTURES ${SUITESPARSE_CUDA_ARCHITECTURES} )
    endif ( )
else ( )
    message ( STATUS "CUDA: not enabled" )
endif ( )

