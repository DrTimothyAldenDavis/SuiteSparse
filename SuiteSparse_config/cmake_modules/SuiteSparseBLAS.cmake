#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseBLAS.cmake
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2023, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# SuiteSparse interface to the Fortran BLAS library.
# cmake 3.22 is required because BLA_SIZEOF_INTEGER is used.

# The Intel MKL BLAS is highly recommended.  It is free to download (but be
# sure to check their license to make sure you accept it).   See:
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.htm

cmake_minimum_required ( VERSION 3.22 )

# To select a specific BLAS: set to the BLA_VENDOR options from FindBLAS.cmake
if ( DEFINED ENV{BLA_VENDOR} )
    set ( BLA_VENDOR $ENV{BLA_VENDOR} )
endif ( )
set ( BLA_VENDOR "ANY" CACHE STRING
    "if ANY (default): searches for any BLAS. Otherwise: search for a specific BLAS" )

# To allow the use of a BLAS with 64-bit integers, set this to true
option ( ALLOW_64BIT_BLAS
    "OFF (default): use only 32-bit BLAS.  ON: look for 32 or 64-bit BLAS" off )

# dynamic/static linking with BLAS
option ( BLA_STATIC
    "OFF (default): dynamic linking of BLAS.  ON: static linking of BLAS" off )

#-------------------------------------------------------------------------------
# look for a specific BLAS library
#-------------------------------------------------------------------------------

# To request specific BLAS, use either (for example):
#
#   CMAKE_OPTIONS="-DBLA_VENDOR=Apple" make
#   cd build ; cmake -DBLA_VENDOR=Apple .. ; make
#
# Use the ALLOW_64BIT_BLAS to select 64-bit or 32-bit BLAS.  This setting is
# strictly enforced.  If set to true, then only a 64-bit BLAS is allowed.
# If this is not found, no 32-bit BLAS is considered, and the build will fail.
#
# If the BLA_VENDOR string implies a 64-bit BLAS, then ALLOW_64BIT_BLAS is set
# to true, ignoring the setting from the user (Intel10_64ilp* and Arm_64ilp*).
#
# The default for ALLOW_64BIT_BLAS is false.

if ( NOT (BLA_VENDOR STREQUAL "ANY" ) )
    # only look for the BLAS from a single vendor
    if ( ( BLA_VENDOR MATCHES "64ilp" ) OR
         ( BLA_VENDOR MATCHES "ilp64" ) )
        # Intel10_64ilp* or Arm_ilp64*
        set ( ALLOW_64BIT_BLAS true )
    endif ( )
    if ( ALLOW_64BIT_BLAS )
        # only look for 64-bit BLAS
        set ( BLA_SIZEOF_INTEGER 8 )
        message ( STATUS "Looking for 64-BLAS: "  ${BLA_VENDOR} )
    else ( )
        # only look for 32-bit BLAS
        message ( STATUS "Looking for 32-BLAS: "  ${BLA_VENDOR} )
        set ( BLA_SIZEOF_INTEGER 4 )
    endif ( )
    find_package ( BLAS REQUIRED )
    if ( BLA_SIZEOF_INTEGER EQUAL 8 )
        include ( SuiteSparseBLAS64 )
        if ( BLA_VENDOR STREQUAL "Intel10_64ilp" )
            add_compile_definitions ( MKL_ILP64 )
        endif ( )
    else ( BLA_SIZEOF_INTEGER EQUAL 4 )
        include ( SuiteSparseBLAS32 )
    endif ( )
    message ( STATUS "Specific BLAS: ${BLA_VENDOR} found: ${BLAS_FOUND}" )
    return ( )
endif ( )

#-------------------------------------------------------------------------------
# Look for any 64-bit BLAS, if allowed
#-------------------------------------------------------------------------------

# If ALLOW_64BIT_BLAS is true, then a 64-bit BLAS is preferred.
# If not found, a 32-bit BLAS is sought (below)

if ( ALLOW_64BIT_BLAS )

    # Look for Intel MKL BLAS with 64-bit integers
    message ( STATUS "Looking for Intel 64-bit BLAS" )
    set ( BLA_VENDOR Intel10_64ilp )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    if ( BLAS_FOUND )
        include ( SuiteSparseBLAS64 )
        add_compile_definitions ( MKL_ILP64 )
        return ( )
    endif ( )

    # Look for ARM BLAS with 64-bit integers
    message ( STATUS "Looking for ARM 64-bit BLAS" )
    set ( BLA_VENDOR Arm_ilp64_mp )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    if ( BLAS_FOUND )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )

    # Look for IBM BLAS with 64-bit integers
    message ( STATUS "Looking for IBM ESSL 64-bit BLAS" )
    set ( BLA_VENDOR IBMESSL_SMP )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    if ( BLAS_FOUND )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )

    # Look for OpenBLAS with 64-bit integers
    message ( STATUS "Looking for 64-bit OpenBLAS" )
    set ( BLA_VENDOR OpenBLAS )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    if ( BLAS_FOUND )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )

    # Look for any 64-bit BLAS
    unset ( BLA_VENDOR )
    message ( STATUS "Looking for any 64-bit BLAS" )
    set ( BLA_SIZEOF_INTEGER 8 )
    find_package ( BLAS )
    if ( BLAS_FOUND )
        include ( SuiteSparseBLAS64 )
        return ( )
    endif ( )

endif ( )

#-------------------------------------------------------------------------------
# Look for a 32-bit BLAS, if no 64-bit BLAS has been found
#-------------------------------------------------------------------------------

# Look for Intel MKL BLAS with 32-bit integers (and 64-bit pointer)
message ( STATUS "Looking for Intel 32-bit BLAS" )
set ( BLA_VENDOR Intel10_64lp )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
if ( BLAS_FOUND )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for Apple Accelerate Framework (32-bit only)
message ( STATUS "Looking for 32-bit Apple BLAS" )
set ( BLA_VENDOR Apple )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
if ( BLAS_FOUND )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for ARM BLAS with 32-bit integers
message ( STATUS "Looking for ARM 32-bit BLAS" )
set ( BLA_VENDOR Arm_mp )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
if ( BLAS_FOUND )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for IBM BLAS with 32-bit integers
message ( STATUS "Looking for IBM ESSL 32-bit BLAS" )
set ( BLA_VENDOR IBMESSL_SMP )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
if ( BLAS_FOUND )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for OpenBLAS with 32-bit integers
message ( STATUS "Looking for 32-bit OpenBLAS" )
set ( BLA_VENDOR OpenBLAS )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
if ( BLAS_FOUND )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

# Look for FLAME BLAS(32-bit only)
message ( STATUS "Looking for 32-bit FLAME (BLIS) BLAS" )
set ( BLA_VENDOR FLAME )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS )
if ( BLAS_FOUND )
    include ( SuiteSparseBLAS32 )
    return ( )
endif ( )

#-------------------------------------------------------------------------------
# do not change the following
#-------------------------------------------------------------------------------

# Look for any 32-bit BLAS (this is required)
unset ( BLA_VENDOR )
message ( STATUS "Looking for any 32-bit BLAS" )
set ( BLA_SIZEOF_INTEGER 4 )
find_package ( BLAS REQUIRED )
include ( SuiteSparseBLAS32 )

