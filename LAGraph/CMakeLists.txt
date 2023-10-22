#-------------------------------------------------------------------------------
# LAGraph/CMakeLists.txt:  cmake script for LAGraph
#-------------------------------------------------------------------------------

# LAGraph, (c) 2019-2023 by The LAGraph Contributors, All Rights Reserved.
# SPDX-License-Identifier: BSD-2-Clause
#
# For additional details (including references to third party source code and
# other files) see the LICENSE file or contact permission@sei.cmu.edu. See
# Contributors.txt for a full list of contributors. Created, in part, with
# funding and support from the U.S. Government (see Acknowledgments.txt file).
# DM22-0790

#-------------------------------------------------------------------------------

# CMakeLists.txt: instructions for cmake to build LAGraph.  An ANSI C11
# compiler is required.  First, install any GraphBLAS library.  Alternatively,
# use ../GraphBLAS (see comments below).
#
# To compile the LAGraph library and its tests and benchmarks, and run the
# tests:
#
#   cd build
#   cmake ..
#   make
#   make test
#
# If that fails for any reason, make sure your compiler supports ANSI C11.  You
# could try changing your compiler for the cmake command, for example:
#
#   CC=icc cmake ..
#   CC=xlc cmake ..
#   CC=gcc cmake ..

#-------------------------------------------------------------------------------
# get the version
#-------------------------------------------------------------------------------

cmake_minimum_required ( VERSION 3.13 )

set ( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    ${CMAKE_SOURCE_DIR}/cmake_modules
    ${CMAKE_SOURCE_DIR}/../cmake_modules
    ${CMAKE_SOURCE_DIR}/../SuiteSparse_config/cmake_modules
    )

cmake_policy ( SET CMP0042 NEW )
cmake_policy ( SET CMP0048 NEW )
set ( CMAKE_MACOSX_RPATH TRUE )

# version of LAGraph
set ( LAGraph_DATE "Aug 2, 2023" )
set ( LAGraph_VERSION_MAJOR 1 )
set ( LAGraph_VERSION_MINOR 0 )
set ( LAGraph_VERSION_SUB   2 )

include ( SuiteSparsePolicy )

project ( lagraph
          VERSION "${LAGraph_VERSION_MAJOR}.${LAGraph_VERSION_MINOR}.${LAGraph_VERSION_SUB}" )

enable_language ( C )

option ( NOPENMP "ON: do not use OpenMP.  OFF (default): use OpenMP" OFF )

# configure LAGraph.h with the LAGraph date and version
configure_file (
    "config/LAGraph.h.in"
    "${PROJECT_SOURCE_DIR}/include/LAGraph.h"
    NEWLINE_STYLE LF )

#-------------------------------------------------------------------------------
# code coverage and build type
#-------------------------------------------------------------------------------

# To compile with test coverage:
#   cd build
#   cmake -DCOVERAGE=1 ..
#   make -j8
#   make test_coverage

if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
	if ( COVERAGE )
                message ( STATUS "============== Code coverage enabled ===============" )
		set ( CMAKE_BUILD_TYPE Debug )
                # On the Mac, you need gcov-11 from homebrew (part of gcc-11):
                # and uncomment this line:
                # set ( GCOV_PATH /usr/local/bin/gcov-11)
		include ( CodeCoverage )

		append_coverage_compiler_flags ( )

		# turn off optimization for non-skewed coverage reports
		set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -DCOVERAGE" )

		setup_target_for_coverage_lcov (
			NAME test_coverage
			EXECUTABLE ctest
			DEPENDENCIES ${PROJECT_NAME}
			BASE_DIRECTORY "."
			NO_DEMANGLE TRUE
                        EXCLUDE "*/benchmark/*" "deps/json*/*"
                            "src/test/include/acutest.h"
		)
	endif ( )
endif ( )

# For development only, not for end-users:
# set ( CMAKE_BUILD_TYPE Debug )

if ( NOT CMAKE_BUILD_TYPE )
    set ( CMAKE_BUILD_TYPE Release )
endif ( )

#-------------------------------------------------------------------------------
# Find the GraphBLAS installation
#-------------------------------------------------------------------------------

# If GraphBLAS is not in a standard installation location, either
#     export GRAPHBLAS_ROOT <path>
# or
#     GRAPHBLAS_ROOT=<path> cmake ..
# or uncomment the next line:
# set ( ENV{GRAPHBLAS_ROOT} ${CMAKE_SOURCE_DIR}/../GraphBLAS )
message ( STATUS "GraphBLAS_ROOT: ${GraphBLAS_ROOT} $ENV{GraphBLAS_ROOT}" )
message ( STATUS "GRAPHBLAS_ROOT: ${GRAPHBLAS_ROOT} $ENV{GRAPHBLAS_ROOT}" )
find_package ( GraphBLAS 8.2.1
    PATHS ${CMAKE_SOURCE_DIR}/../GraphBLAS/build NO_DEFAULT_PATH )
if ( NOT TARGET SuiteSparse::GraphBLAS )
    find_package ( GraphBLAS 8.2.1 REQUIRED )
endif ( )

#-------------------------------------------------------------------------------
# determine what user threading model to use
#-------------------------------------------------------------------------------

if ( COVERAGE )
    message ( STATUS "OpenMP disabled for test coverage" )
else ( )
    if ( NOPENMP )
        set ( OPENMP_C_FOUND OFF )
    else ( )
        find_package ( OpenMP )
        find_package ( Threads )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# report status
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE build type:          " ${CMAKE_BUILD_TYPE} )
message ( STATUS "CMAKE source directory:    " ${CMAKE_SOURCE_DIR} )
message ( STATUS "CMAKE build directory:     " ${CMAKE_BINARY_DIR} )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    message ( STATUS "CMAKE C Flags debug:       " ${CMAKE_C_FLAGS_DEBUG} )
else ( )
    message ( STATUS "CMAKE C Flags release:     " ${CMAKE_C_FLAGS_RELEASE} )
endif ( )

message ( STATUS "CMAKE compiler ID:         " ${CMAKE_C_COMPILER_ID} )
message ( STATUS "CMAKE thread library:      " ${CMAKE_THREAD_LIBS_INIT} )
message ( STATUS "CMAKE have pthreads:       " ${CMAKE_USE_PTHREADS_INIT}  )
message ( STATUS "CMAKE have Win32 pthreads: " ${CMAKE_USE_WIN32_THREADS_INIT} )
message ( STATUS "CMAKE have OpenMP:         " ${OPENMP_C_FOUND} )

#-------------------------------------------------------------------------------
# include directories for LAGraph library
#-------------------------------------------------------------------------------

include_directories ( ${CMAKE_SOURCE_DIR}/include
                      ${CMAKE_SOURCE_DIR}/src/utility
                      ${CMAKE_SOURCE_DIR}/test/include
                      ${GRAPHBLAS_INCLUDE_DIR} )
include_directories ( "/usr/local/include" )

# tell LAGraph where to find its own source (for LAGraph/data files)
set ( CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DLGDIR=${CMAKE_SOURCE_DIR}" )

#-------------------------------------------------------------------------------
# compiler options
#-------------------------------------------------------------------------------

# check which compiler is being used.  If you need to make
# compiler-specific modifications, here is the place to do it.
if ( "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    # check all warnings:
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Werror " )
    # set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -g " )
    #if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 4.9 )
    #    message ( FATAL_ERROR "gcc version must be at least 4.9" )
    #endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "Intel" )
    # note that -g can be used, for VTune.  Comment out the following line
    # to compile without -g.
    # set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -g " )
    # check all warnings:
#   set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -w3 " )
    #if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 18.0 )
    #    message ( FATAL_ERROR "icc version must be at least 18.0" )
    #endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "Clang" )
    # options for clang
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -O3 " )
    #if ( CMAKE_C_COMPILER_VERSION VERSION_LESS 3.3 )
    #    message ( FATAL_ERROR "clang version must be at least 3.3" )
    #endif ( )
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC" )
    # options for MicroSoft Visual Studio
elseif ( "${CMAKE_C_COMPILER_ID}" STREQUAL "PGI" )
    # options for PGI pgcc compiler
    set ( CMAKE_C_FLAGS    "${CMAKE_C_FLAGS} -acc -Minfo=accel -Mcuda -Mnoopenmp -noswitcherror -fPIC " )
endif ( )

if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_DEBUG}" )
else ( )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_RELEASE}" )
endif ( )

#-------------------------------------------------------------------------------
# force use of vanilla code
#-------------------------------------------------------------------------------

if ( LAGRAPH_VANILLA )
    set ( CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} -DLAGRAPH_VANILLA=1 " )
    message ( STATUS "Vanilla build: not relying on any GxB extensions" )
else ( )
    message ( STATUS "GxB build: relying on SuiteSparse GxB extensions" )
endif ( )

#-------------------------------------------------------------------------------
# print final C flags
#-------------------------------------------------------------------------------

message ( STATUS "CMAKE C flags: " ${CMAKE_C_FLAGS} )

#-------------------------------------------------------------------------------
# enable testing and add subdirectories
#-------------------------------------------------------------------------------

# allow ctest to find the binaries:
set ( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )
set ( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR} )

enable_testing ( )

add_subdirectory ( src )
add_subdirectory ( experimental )
if ( DEFINED ENV{READTHEDOCS} )
    add_subdirectory ( rtdocs )
endif ( )

#-------------------------------------------------------------------------------
# LAGraph installation location
#-------------------------------------------------------------------------------

# install in ${CMAKE_INSTALL_PREFIX}/lib and ${CMAKE_INSTALL_PREFIX}/include.
# Requires "sudo make install" if this is /usr/local (default).

message ( STATUS "Installation in: ${CMAKE_INSTALL_PREFIX}" )
include ( GNUInstallDirs )

install ( TARGETS lagraph lagraphx
    LIBRARY DESTINATION ${SUITESPARSE_LIBDIR}
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    RUNTIME DESTINATION ${SUITESPARSE_BINDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

install ( TARGETS lagraph_static lagraphx_static
    ARCHIVE DESTINATION ${SUITESPARSE_LIBDIR}
    PUBLIC_HEADER DESTINATION ${SUITESPARSE_INCLUDEDIR} )

