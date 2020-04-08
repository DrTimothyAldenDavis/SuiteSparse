//------------------------------------------------------------------------------
// GB_printf.c: printing for GraphBLAS *check functions
//------------------------------------------------------------------------------

// SuiteSparse:GraphBLAS, Timothy A. Davis, (c) 2017-2020, All Rights Reserved.
// http://suitesparse.com   See GraphBLAS/Doc/License.txt for license.

//------------------------------------------------------------------------------

#include "GB.h"

int (* GB_printf_function ) (const char *format, ...) = NULL ;
int (* GB_flush_function  ) ( void ) = NULL ;

