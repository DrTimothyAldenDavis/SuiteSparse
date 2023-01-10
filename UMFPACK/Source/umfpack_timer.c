//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_timer: timing routine
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Returns the time in seconds used by the process.  BE
    CAREFUL:  if you compare the run time of UMFPACK with other sparse matrix
    packages, be sure to use the same timer.  See umfpack.h for details.
    See umfpack_tictoc.h, which is the timer used internally by UMFPACK.
*/

#include "umfpack.h"

double umfpack_timer ( void )
{
    return (SuiteSparse_time ( )) ;
}
