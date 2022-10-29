#-------------------------------------------------------------------------------
# SuiteSparse/SuiteSparse_config/cmake_modules/SuiteSparseBLAS32.cmake
#-------------------------------------------------------------------------------

# SuiteSparse_config, Copyright (c) 2012-2022, Timothy A. Davis.
# All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# actions taken when a 32-bit BLAS has been found

message ( STATUS "Found 32-bit BLAS+LAPACK" )
set ( SuiteSparse_BLAS_integer "int32_t" )

