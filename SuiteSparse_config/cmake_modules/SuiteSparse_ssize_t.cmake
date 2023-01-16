#-------------------------------------------------------------------------------
# SuiteSparse/cmake_modules/SuiteSparse_ssize_t.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: BSD-3-clause

#-------------------------------------------------------------------------------

# determine if the compiler defines ssize_t

include ( CheckCSourceCompiles )

set ( ssize_t_source
"   #include <sys/types.h>
    int main (void)
    {
        ssize_t x = 0 ;
        return (0) ;
    }
" )

check_c_source_compiles ( "${ssize_t_source}" TEST_FOR_SSIZE_T )

if ( TEST_FOR_SSIZE_T )
    set ( HAVE_SSIZE_T true )
    message ( STATUS "#include <sys/types.h> and ssize_t: OK" )
else ( )
    set ( HAVE_SSIZE_T false )
    message ( STATUS "#include <sys/types.h> and ssize_t: not found" )
endif ( )

