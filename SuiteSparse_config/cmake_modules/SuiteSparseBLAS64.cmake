#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseBLAS64.cmake
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# actions taken when a 64-bit BLAS has been found

message ( STATUS "Found ${BLA_VENDOR} 64-bit BLAS+LAPACK" )
add_compile_definitions ( BLAS_${BLA_VENDOR} )
add_compile_definitions ( BLAS64 )
set ( SuiteSparse_BLAS_integer "int64_t" )

#-------------------------------------------------------------------------------
# Examine the suffix appended to the Fortran 64-bit BLAS+LAPACK functions
#-------------------------------------------------------------------------------

# OpenBLAS can be compiled by appending a suffix to each routine, so that the
# Fortan routine dgemm becomes dgemm_64, which denotes a version of dgemm with
# 64-bit integer parameters.  The Sun Performance library does the same thing.

# If the suffix does not contain "_", use (Sun Perf., for example):

#      cd build ; cmake -DBLAS64_SUFFIX="64" ..

# If the suffix contains "_" (OpenBLAS in spack for example), use the
# following:

#      cd build ; cmake -DBLAS64_SUFFIX="_64" ..

# This setting could be used by the spack packaging of SuiteSparse when linked
# with the spack-installed OpenBLAS with 64-bit integers.  See
# https://github.com/spack/spack/blob/develop/var/spack/repos/builtin/packages/suite-sparse/package.py

if ( DEFINED BLAS64_SUFFIX )
    # append BLAS64_SUFFIX to each BLAS and LAPACK name
    string ( FIND ${BLAS64_SUFFIX} "_" HAS_UNDERSCORE )
    message ( STATUS "BLAS64_suffix: ${BLAS64_SUFFIX}" )
    if ( HAS_UNDERSCORE EQUAL -1 )
        message ( STATUS "BLAS64 suffix has no underscore" )
        add_compile_definitions ( BLAS64_SUFFIX=${BLAS64_SUFFIX} )
    else ( )
        message ( STATUS "BLAS64 suffix has an underscore" )
        add_compile_definitions ( BLAS64__SUFFIX=${BLAS64_SUFFIX} )
    endif ( )
endif ( )

