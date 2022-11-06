#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseBLAS32.cmake
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# actions taken when a 32-bit BLAS has been found

message ( STATUS "Found ${BLA_VENDOR} 32-bit BLAS+LAPACK" )
add_compile_definitions ( BLAS_${BLA_VENDOR} )
set ( SuiteSparse_BLAS_integer "int32_t" )

