#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/SuiteSparseReport.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2012-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------
# report status and compile flags
#-------------------------------------------------------------------------------

message ( STATUS "------------------------------------------------------------------------" )
message ( STATUS "SuiteSparse CMAKE report for: ${PROJECT_NAME}" )
message ( STATUS "------------------------------------------------------------------------" )
if ( NOT SUITESPARSE_ROOT_CMAKELISTS )
    message ( STATUS "inside common SuiteSparse root:  ${INSIDE_SUITESPARSE}" )
    message ( STATUS "install in SuiteSparse/lib and SuiteSparse/include: ${SUITESPARSE_LOCAL_INSTALL}" )
endif ( )
message ( STATUS "build type:           ${CMAKE_BUILD_TYPE}" )
message ( STATUS "BUILD_SHARED_LIBS:    ${BUILD_SHARED_LIBS}" )
message ( STATUS "BUILD_STATIC_LIBS:    ${BUILD_STATIC_LIBS}" )
message ( STATUS "C compiler:           ${CMAKE_C_COMPILER} ")
message ( STATUS "C flags:              ${CMAKE_C_FLAGS}" )
message ( STATUS "C++ compiler:         ${CMAKE_CXX_COMPILER}" )
message ( STATUS "C++ flags:            ${CMAKE_CXX_FLAGS}" )
if ( ${CMAKE_BUILD_TYPE} STREQUAL "Debug" )
    message ( STATUS "C Flags debug:        ${CMAKE_C_FLAGS_DEBUG} ")
    message ( STATUS "C++ Flags debug:      ${CMAKE_CXX_FLAGS_DEBUG} ")
else ( )
    message ( STATUS "C Flags release:      ${CMAKE_C_FLAGS_RELEASE} ")
    message ( STATUS "C++ Flags release:    ${CMAKE_CXX_FLAGS_RELEASE} ")
endif ( )
if ( SUITESPARSE_HAS_FORTRAN )
    message ( STATUS "Fortran compiler:     ${CMAKE_Fortran_COMPILER} " )
else ( )
    message ( STATUS "Fortran compiler:     none" )
endif ( )
get_property ( CDEFN DIRECTORY PROPERTY COMPILE_DEFINITIONS )
message ( STATUS "compile definitions:  ${CDEFN}")
if ( DEFINED SuiteSparse_BLAS_integer )
    message ( STATUS "BLAS integer:         ${SuiteSparse_BLAS_integer}" )
endif ( )
if ( DEFINED CMAKE_CUDA_ARCHITECTURES )
    message ( STATUS "CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}" )
endif ( )
message ( STATUS "------------------------------------------------------------------------" )
