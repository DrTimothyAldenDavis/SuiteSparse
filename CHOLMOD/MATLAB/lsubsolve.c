//------------------------------------------------------------------------------
// CHOLMOD/MATLAB/lsubsolve: MATLAB interface to CHOLMOD solver
//------------------------------------------------------------------------------

// CHOLMOD/MATLAB Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// [x xset] = lsubsolve (L,kind,P,b,system)
//
// L is a sparse lower triangular matrix, of size n-by-n.  kind = 0 if
// L is from an LL' factorization, 1 if from LDL' (in which case L contains
// L in the lower part and D on the diagonal).  P is a permutation vector, or
// empty (which means P is identity).  b is a sparse column vector.
// system is a number between 0 and 8:
//
// Given L or LD, a permutation P, and a sparse right-hand size b,
// solve one of the following systems:
//
//      Ax=b        0: CHOLMOD_A        also applies the permutation L->Perm
//      LDL'x=b     1: CHOLMOD_LDLt     does not apply L->Perm
//      LDx=b       2: CHOLMOD_LD
//      DL'x=b      3: CHOLMOD_DLt
//      Lx=b        4: CHOLMOD_L
//      L'x=b       5: CHOLMOD_Lt
//      Dx=b        6: CHOLMOD_D
//      x=Pb        7: CHOLMOD_P        apply a permutation (P is L->Perm)
//      x=P'b       8: CHOLMOD_Pt       apply an inverse permutation
//
// The solution x is a dense vector, but it is a subset of the entire solution,
// x is zero except where xset is 1.  xset is reach of b (or P*b) in the graph
// of L.  If P is empty then it is treated as the identity permutation.
//
// No zeros can be dropped from the stucture of the Cholesky factorization
// because this function gets its elimination tree from L itself.  See ldlchol
// for a method of constructing a sparse L with explicit zeros that are
// normally dropped by MATLAB.
//
// This function is only meant for testing the cholmod_solve2 function.  The
// cholmod_solve2 function takes O(flops) time, but the setup time in this
// mexFunction wrapper can dominate that time.

#include "sputil2.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0, *Px ;
    int64_t *Lp, *Lnz, *Xp, *Xi, xnz, *Perm, *Lprev, *Lnext ;
    cholmod_sparse *Bset, Bmatrix, *Xset ;
    cholmod_dense *Bdense, *X, *Y, *E ;
    cholmod_factor *L ;
    cholmod_common Common, *cm ;
    int64_t k, j, n, head, tail, xsetlen ;
    int sys, kind ;

    //--------------------------------------------------------------------------
    // start CHOLMOD and set parameters
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;
    sputil2_config (SPUMONI, cm) ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (nargin != 5 || nargout > 2)
    {
        mexErrMsgTxt ("usage: [x xset] = lsubsolve (L,kind,P,b,system)") ;
    }

    n = mxGetN (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || n != mxGetM (pargin [0]))
    {
        mexErrMsgTxt ("lsubsolve: L must be sparse and square") ;
    }
    if (mxGetNumberOfElements (pargin [1]) != 1)
    {
        mexErrMsgTxt ("lsubsolve: kind must be a scalar") ;
    }

    if (mxIsSparse (pargin [2]) ||
       !(mxIsEmpty (pargin [2]) || mxGetNumberOfElements (pargin [2]) == n))
    {
        mexErrMsgTxt ("lsubsolve: P must be size n, or empty") ;
    }

    if (mxGetM (pargin [3]) != n || mxGetN (pargin [3]) != 1)
    {
        mexErrMsgTxt ("lsubsolve: b wrong dimension") ;
    }
    if (!mxIsSparse (pargin [3]))
    {
        mexErrMsgTxt ("lxbpattern: b must be sparse") ;
    }
    if (mxGetNumberOfElements (pargin [4]) != 1)
    {
        mexErrMsgTxt ("lsubsolve: system must be a scalar") ;
    }

    //--------------------------------------------------------------------------
    // get the inputs
    //--------------------------------------------------------------------------

    kind = (int) mxGetScalar (pargin [1]) ;
    sys  = (int) mxGetScalar (pargin [4]) ;

    //--------------------------------------------------------------------------
    // get the sparse b
    //--------------------------------------------------------------------------

    // get sparse matrix B (unsymmetric)
    size_t Bset_xsize = 0 ;
    Bset = sputil2_get_sparse (pargin [3], 0, CHOLMOD_DOUBLE, &Bmatrix,
        &Bset_xsize, cm) ;

    Bdense = cholmod_l_sparse_to_dense (Bset, cm) ;
    Bset->x = NULL ;
    Bset->z = NULL ;
    Bset->xtype = CHOLMOD_PATTERN ;

    //--------------------------------------------------------------------------
    // construct a shallow copy of the input sparse matrix L
    //--------------------------------------------------------------------------

    // the construction of the CHOLMOD takes O(n) time and memory

    // allocate the CHOLMOD symbolic L
    L = cholmod_l_allocate_factor (n, cm) ;
    L->ordering = CHOLMOD_NATURAL ;

    // get the MATLAB L
    L->p = mxGetJc (pargin [0]) ;
    L->i = mxGetIr (pargin [0]) ;
    L->x = mxGetData (pargin [0]) ;
    L->z = NULL ;

    // allocate and initialize the rest of L
    L->nz = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
    Lp = L->p ;
    Lnz = L->nz ;
    for (j = 0 ; j < n ; j++)
    {
        Lnz [j] = Lp [j+1] - Lp [j] ;
    }

    // these pointers are not accessed in cholmod_solve2
    L->prev = cholmod_l_malloc (n+2, sizeof (int64_t), cm) ;
    L->next = cholmod_l_malloc (n+2, sizeof (int64_t), cm) ;
    Lprev = L->prev ;
    Lnext = L->next ;

    head = n+1 ;
    tail = n ;
    Lnext [head] = 0 ;
    Lprev [head] = -1 ;
    Lnext [tail] = -1 ;
    Lprev [tail] = n-1 ;
    for (j = 0 ; j < n ; j++)
    {
        Lnext [j] = j+1 ;
        Lprev [j] = j-1 ;
    }
    Lprev [0] = head ;

    L->xtype = (mxIsComplex (pargin [0])) ? CHOLMOD_COMPLEX : CHOLMOD_REAL ;
    L->nzmax = Lp [n] ;

    // get the permutation
    if (mxIsEmpty (pargin [2]))
    {
        L->Perm = NULL ;
        Perm = NULL ;
    }
    else
    {
        // convert from double to int64_t, and from 1-based to 0-based
        L->ordering = CHOLMOD_GIVEN ;
        L->Perm = cholmod_l_malloc (n, sizeof (int64_t), cm) ;
        Perm = L->Perm ;
        Px = (double *) mxGetData (pargin [2]) ;
        for (k = 0 ; k < n ; k++)
        {
            Perm [k] = ((int64_t) Px [k]) - 1 ;
        }
    }

    // set the kind, LL' or LDL'
    L->is_ll = (kind == 0) ;

    //--------------------------------------------------------------------------
    // solve the system
    //--------------------------------------------------------------------------

    X = cholmod_l_zeros (n, 1, L->xtype, cm) ;
    Xset = NULL ;
    Y = NULL ;
    E = NULL ;

    cholmod_l_solve2 (sys, L, Bdense, Bset, &X, &Xset, &Y, &E, cm) ;

    cholmod_l_free_dense (&Y, cm) ;
    cholmod_l_free_dense (&E, cm) ;

    //--------------------------------------------------------------------------
    // return result
    //--------------------------------------------------------------------------

    pargout [0] = sputil2_put_dense (&X, mxDOUBLE_CLASS, cm) ;

    // fill numerical values of Xset with one's
    if (Xset == NULL)
    {
        // Px=b and P'x=b do not return Xset.  Create Xset = 0:n-1
        Xset = cholmod_l_spzeros (n, 1, n, CHOLMOD_REAL, cm) ;
        int64_t *Xsetp = Xset->p ;
        int64_t *Xseti = Xset->i ;
        double  *Xsetx = Xset->x ;
        Xsetp [0] = 0 ;
        Xsetp [1] = n ;
        for (k = 0 ; k < n ; k++)
        {
            Xseti [k] = k ;
            Xsetx [k] = 1 ;
        }
    }
    else
    {
        // Xset is returned, but needs to be converted from PATTERN to REAL
        int64_t *Xsetp = Xset->p ;
        xsetlen = Xsetp [1] ;
        Xset->x = cholmod_l_malloc (xsetlen, sizeof (double), cm) ;
        double *Xsetx = Xset->x ;
        for (k = 0 ; k < xsetlen ; k++)
        {
            Xsetx [k] = 1 ;
        }
        Xset->xtype = CHOLMOD_REAL ;
    }

    pargout [1] = sputil2_put_sparse (&Xset, mxDOUBLE_CLASS,
        /* Xset is binary so it has no zeros to drop */ false, cm) ;

    //--------------------------------------------------------------------------
    // free workspace and the CHOLMOD L, except for what is copied to MATLAB
    //--------------------------------------------------------------------------

    L->p = NULL ;
    L->i = NULL ;
    L->x = NULL ;
    L->z = NULL ;
    cholmod_l_free_factor (&L, cm) ;
    sputil2_free_sparse (&Bset, &Bmatrix, Bset_xsize, cm) ;
    cholmod_l_finish (cm) ;
    if (SPUMONI > 0) cholmod_l_print_common (" ", cm) ;
}

