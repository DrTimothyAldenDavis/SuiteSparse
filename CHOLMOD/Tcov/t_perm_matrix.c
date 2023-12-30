//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_perm_matrix: create a permutation matrix from perm vector
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// if perm is NULL, the identity permutation matrix is returned

cholmod_sparse *perm_matrix (Int *perm, Int n, int xdtype,
    cholmod_common *Common)
{

    // P = I
    cholmod_sparse *P = CHOLMOD(speye) (n, n, xdtype, Common) ;

    // copy perm [0..n-1] as the row indices of P 
    if (P != NULL && perm != NULL)
    {
        Int *Pi = P->i ;
        for (Int j = 0 ; j < n ; j++)
        {
            Pi [j] = perm [j] ;
        }
    }

    return (P) ;
}

