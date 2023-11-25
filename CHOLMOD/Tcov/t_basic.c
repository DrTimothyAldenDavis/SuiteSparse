//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_basic: basic tests
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "cm.h"

void basic1 (cholmod_common *cm)
{

    //--------------------------------------------------------------------------
    // hypot and divcomplex
    //--------------------------------------------------------------------------

    double maxerr = 0 ;
    double ax = 4.3 ;
    double az = 9.2 ;
    double bx, bz, cx, cz ;
    int bzero ;

    for (Int i = 0 ; i <= 1 ; i++)
    {
        if (i == 0)
        {
            bx = 3.14159 ;
            bz = -1.2 ;
        }
        else
        {
            bx = 0.9 ;
            bz = -1.2 ;
        }

        // c = a/b
        bzero = CHOLMOD(divcomplex)(ax, az, bx, bz, &cx, &cz) ;
        OK (!bzero) ;
        // d = c*b
        double dx = cx * bx - cz * bz ;
        double dz = cz * bx + cx * bz ;
        // e = d-a, which should be zero
        double fx = dx - ax ;
        double fz = dz - az ;
        // r = hypot (fx,fz)
        double r = CHOLMOD(hypot)(fx, fz) ;
        MAXERR (maxerr, r, 1) ;
        OK (r < 1e-14) ;

        // c = a/b using SuiteSparse_divcomplex
        cx = 0 ;
        cz = 0 ;
        bzero = SuiteSparse_divcomplex (ax, az, bx, bz, &cx, &cz) ;
        OK (!bzero) ;
        // d = c*b
        dx = cx * bx - cz * bz ;
        dz = cz * bx + cx * bz ;
        // e = d-a, which should be zero
        fx = dx - ax ;
        fz = dz - az ;
        // r = hypot (fx,fz)
        r = SuiteSparse_hypot (fx, fz) ;
        MAXERR (maxerr, r, 1) ;
        OK (r < 1e-14) ;
    }

    printf ("basic1: %g\n", maxerr) ;

    // divide by zero
    bx = 0 ;
    bz = 0 ;
    bzero = CHOLMOD(divcomplex)(ax, az, bx, bz, &cx, &cz) ;
    OK (bzero) ;
    printf ("c: %g %g\n", cx, cz) ;
    OK (isinf (cx)) ;
    OK (isinf (cz)) ;

    bzero = SuiteSparse_divcomplex (ax, az, bx, bz, &cx, &cz) ;
    OK (bzero) ;
    printf ("c: %g %g\n", cx, cz) ;
    OK (isinf (cx)) ;
    OK (isinf (cz)) ;

    // zero divided by zero
    ax = 0 ;
    az = 0 ;

    bzero = CHOLMOD(divcomplex)(ax, az, bx, bz, &cx, &cz) ;
    OK (bzero) ;
    printf ("c: %g %g\n", cx, cz) ;
    OK (isnan (cx)) ;

    bzero = SuiteSparse_divcomplex (ax, az, bx, bz, &cx, &cz) ;
    OK (bzero) ;
    printf ("c: %g %g\n", cx, cz) ;
    OK (isnan (cx)) ;

    //--------------------------------------------------------------------------
    // descendant score
    //--------------------------------------------------------------------------

    struct cholmod_descendant_score_t s1, s2 ;
    s1.d = 0 ;
    s2.d = 0 ;
    for (int k1 = -4 ; k1 < 4 ; k1++)
    {
        s1.score = (double) k1 ;
        for (int k2 = -4 ; k2 < 4 ; k2++)
        {
            s2.score = (double) k2 ;
            int result = CHOLMOD(score_comp) (&s1, &s2) ;
            OK (result == (k1 < k2) ? 1 : -1) ;
        }
    }
}

