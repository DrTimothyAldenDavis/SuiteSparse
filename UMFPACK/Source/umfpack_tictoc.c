//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_tictoc: timing routines
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Returns the current wall clock time.  BE CAREFUL:  if you
    compare the run time of UMFPACK with other sparse matrix packages, be sure
    to use the same timer.  See umfpack.h for details.
*/

#include "umf_internal.h"

void umfpack_tic (double stats [2])
{
    stats [0] = SuiteSparse_time ( ) ;
    stats [1] = stats [0] ;
}

void umfpack_toc (double stats [2])
{
    stats [0] = SuiteSparse_time ( ) - stats [0] ;
    stats [1] = stats [0] ;
}
