#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseBLAS.cmake
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# SuiteSparse interface to the BLAS and LAPACK libraries.

cmake_minimum_required ( VERSION 3.22 )

message ( STATUS "BLAS64_suffix:   ${BLAS64_SUFFIX}" )
message ( STATUS "Looking for 64-bit BLAS" )
set ( BLA_SIZEOF_INTEGER 8 )    # requires cmake 3.22 or later
find_package ( BLAS )
find_package ( LAPACK )
if ( BLAS_FOUND AND LAPACK_FOUND )
    # 64-bit BLAS found
    message ( STATUS "Found 64-bit BLAS/LAPACK" )
    add_compile_definitions ( BLAS64 ) 
    set ( SuiteSparse_BLAS_integer "int64_t" )
    if (NOT ${BLAS64_SUFFIX} STREQUAL "")
        # append BLAS64_SUFFIX to each BLAS and LAPACK name
        string ( FIND ${BLAS64_SUFFIX} "_" HAS_UNDERSCORE )
        if ( HAS_UNDERSCORE EQUAL -1 )
            message ( STATUS " BLAS64 suffix has an underscore" )
            add_compile_definitions ( BLAS64__SUFFIX=${BLAS64_SUFFIX} ) 
        else ( )
            message ( STATUS " BLAS64 suffix has no underscore" )
            add_compile_definitions ( BLAS64_SUFFIX=${BLAS64_SUFFIX} ) 
        endif ( )
    endif ( )
else ( ) 
    # look for 32-bit BLAS
    message ( STATUS "Looking for 32-bit BLAS/LAPACK" )
    set ( BLA_SIZEOF_INTEGER 4 )    # requires cmake 3.22 or later
    find_package ( BLAS REQUIRED )
    find_package ( LAPACK REQUIRED )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        message ( STATUS "Found 32-bit BLAS/LAPACK" )
    endif ( )
    set ( SuiteSparse_BLAS_integer "int32_t" )
endif ( ) 

include ( FortranCInterface )

