//------------------------------------------------------------------------------
// UMFPACK/Demo/umfpack_simple: simple demo for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2022, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include <stdio.h>
#include "umfpack.h"

int    n = 5 ;
int    Ap [ ] = {0, 2, 5, 9, 10, 12} ;
int    Ai [ ] = { 0,  1,  0,   2,  4,  1,  2,  3,   4,  2,  1,  4} ;
double Ax [ ] = {2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.} ;
double b [ ] = {8., 45., -3., 3., 19.} ;
double x [5] ;

int main (void)
{
    printf ("%s\n", UMFPACK_VERSION) ;
    printf ("%s", UMFPACK_COPYRIGHT) ;
    printf ("%s", UMFPACK_LICENSE_PART1) ;
    printf ("BLAS used: %s\n", SuiteSparse_BLAS_library ( )) ;
    printf ("BLAS integer size: %d bytes\n",
        (int) sizeof (SUITESPARSE_BLAS_INT)) ;

    double *null = (double *) NULL ;
    int i ;
    void *Symbolic, *Numeric ;
    (void) umfpack_di_symbolic (n, n, Ap, Ai, Ax, &Symbolic, null, null) ;
    (void) umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric, null, null) ;
    umfpack_di_free_symbolic (&Symbolic) ;
    (void) umfpack_di_solve (UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, null, null) ;
    umfpack_di_free_numeric (&Numeric) ;
    for (i = 0 ; i < n ; i++) printf ("x [%d] = %g\n", i, x [i]) ;
    return (0) ;
}

