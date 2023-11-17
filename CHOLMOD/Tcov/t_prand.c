//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_prand: random permutation vector
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// allocate and construct a random permutation of 0:n-1

Int *prand (Int n)
{
    Int *P ;

    P = CHOLMOD(malloc) (n, sizeof (Int), cm) ;
    if (P == NULL)
    {
        return (NULL) ;
    }

    for (Int k = 0 ; k < n ; k++)
    {
        P [k] = k ;
    }

    for (Int k = 0 ; k < n-1 ; k++)
    {
        Int j = k + nrand (n-k) ;
        Int t = P [j] ;
        P [j] = P [k] ;
        P [k] = t ;
    }
    return (P) ;
}

