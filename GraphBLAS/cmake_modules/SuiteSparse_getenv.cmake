#-------------------------------------------------------------------------------
# GraphBLAS/cmake_modules/SuiteSparse_getenv.cmake
#-------------------------------------------------------------------------------

# Copyright (c) 2017-2023, Timothy A. Davis.  All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

#-------------------------------------------------------------------------------

# determine if the compiler and OS support getenv ("HOME")

include ( CheckCSourceRuns )

set ( getenv_src
"   #include <stdio.h>
    #include <stdlib.h>
    int main (void)
    {
        char *home = getenv (\"HOME\") ;
        printf (\"home: %s\", home) ;
        if (home == NULL) return (-1) ;
        return (0) ;
    }
" )

check_c_source_runs ( "${getenv_src}" HAVE_GETENV_HOME )

if ( HAVE_GETENV_HOME )
    message ( STATUS "getenv(\"HOME\"): available" )
else ( )
    add_compile_definitions ( NGETENV_HOME )
    message ( STATUS "getenv(\"HOME\"): not available" )
endif ( )

