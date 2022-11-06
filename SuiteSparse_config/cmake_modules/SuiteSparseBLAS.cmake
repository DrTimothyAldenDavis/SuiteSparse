#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseBLAS.cmake
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# SuiteSparse interface to the Fortran BLAS+LAPACK libraries.
# cmake 3.22 is required because BLA_SIZEOF_INTEGER is used.

# The Intel MKL BLAS is highly recommended.  It is free to download (but be
# sure to check their license to make sure you accept it).   See:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.htm

cmake_minimum_required ( VERSION 3.22 )
include ( FortranCInterface )

# To allow the use of a BLAS with 64-bit integers, set this to true
if ( NOT DEFINED ALLOW_64BIT_BLAS )
    set ( ALLOW_64BIT_BLAS false )
endif ( )

# Look for user specified BLAS - caller is responsible for setting appropriate
# BLAS_* definition
if ( BLA_VENDOR )
    message ( STATUS "Looking for " ${BLA_VENDOR} )
    find_package ( BLAS )
    find_package ( LAPACK )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        if ( BLA_SIZEOF_INTEGER STREQUAL "8" )
            include ( SuiteSparseBLAS64 )
        else ()
            include ( SuiteSparseBLAS32 )
        endif ()
        return ( )
    endif ( )
endif ( )

#-------------------------------------------------------------------------------
# these blocks of code can be rearranged to change the search order for the BLAS
#-------------------------------------------------------------------------------

# Look for Intel MKL BLAS with 64-bit integers
if ( ALLOW_64BIT_BLAS )
    message ( STATUS "Looking for Intel 64-bit BLAS+LAPACK" )
    set ( BLA_VENDOR Intel10_64ilp )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    find_package ( LAPACK )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        add_compile_definitions ( BLAS_INTEL_64ILP )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )
endif ( )

# Look for Intel MKL BLAS with 32-bit integers (and 64-bit pointer)
message ( STATUS "Looking for Intel 32-bit BLAS+LAPACK" )
set ( BLA_VENDOR Intel10_64lp )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
find_package ( LAPACK )
if ( BLAS_FOUND AND LAPACK_FOUND )
    add_compile_definitions ( BLAS_INTEL_64LP )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for Apple Accelerate Framework (32-bit only)
message ( STATUS "Looking for 32-bit Apple BLAS+LAPACK" )
set ( BLA_VENDOR Apple )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
find_package ( LAPACK )
if ( BLAS_FOUND AND LAPACK_FOUND )
    add_compile_definitions ( BLAS_APPLE )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for ARM BLAS with 64-bit integers
if ( ALLOW_64BIT_BLAS )
    message ( STATUS "Looking for ARM 64-bit BLAS+LAPACK" )
    set ( BLA_VENDOR Arm_mp )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    find_package ( LAPACK )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        add_compile_definitions ( BLAS_ARM_MP )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )
endif ( )

# Look for ARM BLAS with 32-bit integers
message ( STATUS "Looking for ARM 32-bit BLAS+LAPACK" )
set ( BLA_VENDOR Arm_ilp64_mp )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
find_package ( LAPACK )
if ( BLAS_FOUND AND LAPACK_FOUND )
    add_compile_definitions ( BLAS_ARM_ILP64_MP )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for IBM BLAS with 64-bit integers
if ( ALLOW_64BIT_BLAS )
    message ( STATUS "Looking for IBM ESSL 64-bit BLAS+LAPACK" )
    set ( BLA_VENDOR IBMESSL_SMP )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    find_package ( LAPACK )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        add_compile_definitions ( BLAS_IBMESSL_SMP_64 )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )
endif ( )

# Look for IBM BLAS with 32-bit integers
message ( STATUS "Looking for IBM ESSL 32-bit BLAS+LAPACK" )
set ( BLA_VENDOR IBMESSL_SMP )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
find_package ( LAPACK )
if ( BLAS_FOUND AND LAPACK_FOUND )
    add_compile_definitions ( BLAS_IBMESSL_SMP_32 )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for OpenBLAS with 64-bit integers
if ( ALLOW_64BIT_BLAS )
    message ( STATUS "Looking for 64-bit OpenBLAS (and LAPACK)" )
    set ( BLA_VENDOR OpenBLAS )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    find_package ( LAPACK )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        add_compile_definitions ( BLAS_OPENBLAS_64 )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )
endif ( )

# Look for OpenBLAS with 32-bit integers
message ( STATUS "Looking for 32-bit OpenBLAS (and LAPACK)" )
set ( BLA_VENDOR OpenBLAS )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
find_package ( LAPACK )
if ( BLAS_FOUND AND LAPACK_FOUND )
    add_compile_definitions ( BLAS_OPENBLAS )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

#-------------------------------------------------------------------------------
# do not change the following
#-------------------------------------------------------------------------------

unset ( BLA_VENDOR )

# Look for any 64-bit BLAS
if ( ALLOW_64BIT_BLAS )
    message ( STATUS "Looking for any 64-bit BLAS+LAPACK" )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    find_package ( LAPACK )
    if ( BLAS_FOUND AND LAPACK_FOUND )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )
endif ( )

# Look for any 32-bit BLAS (this is required)
unset ( BLA_VENDOR )
message ( STATUS "Looking for any 32-bit BLAS+LAPACK" )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS REQUIRED )
find_package ( LAPACK REQUIRED )
include ( SuiteSparseBLAS32 )

