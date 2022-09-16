//------------------------------------------------------------------------------
// CAMD/Demo/camd_simple.c: simple demo program for CAMD
//------------------------------------------------------------------------------

// CAMD, Copyright (c) 2007-2022, Timothy A. Davis, Yanqing Chen, Patrick R.
// Amestoy, and Iain S. Duff.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

#include <stdio.h>
#include "camd.h"

int32_t n = 5 ;
int32_t Ap [ ] = { 0,   2,       6,       10,  12, 14} ;
int32_t Ai [ ] = { 0,1, 0,1,2,4, 1,2,3,4, 2,3, 1,4   } ;
int32_t C [ ] = { 2, 0, 0, 0, 1 } ;
int32_t P [5] ;

int main (void)
{
    int32_t k ;
    (void) camd_order (n, Ap, Ai, P, (double *) NULL, (double *) NULL, C) ;
    for (k = 0 ; k < n ; k++) printf ("P [%d] = %d\n", k, P [k]) ;
    return (0) ;
}

