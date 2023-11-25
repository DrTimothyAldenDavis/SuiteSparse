//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_dump: debug routines for CHOLMOD test programs
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// sparse_dump
//------------------------------------------------------------------------------

// dump a sparse matrix in triplet form.  The up, lo, up/lo format is ignored.

void sparse_dump (cholmod_sparse *A, char *filename, cholmod_common *cm)
{
    if (!A) return ;
    FILE *ff = fopen (filename, "w") ;
    if (!ff) return ;
    Int *Ap = A->p ;
    Int *Ai = A->i ;
    Real *Ax = A->x ;
    Real *Az = A->z ;
    Int *Anz = A->nz ;
    bool packed = A->packed ;
    for (Int j = 0 ; j < A->ncol ; j++)
    {
        Int p = Ap [j] ;
        Int pend = (packed) ? (Ap [j+1]) : (p + Anz [j]) ;
        for ( ; p < pend ; p++)
        {
            Int i = Ai [p] ;
            if (A->xtype == CHOLMOD_PATTERN)
            {
                fprintf (ff, ID " " ID "\n", i, j) ;
            }
            else if (A->xtype == CHOLMOD_REAL)
            {
                fprintf (ff, ID " " ID " %30.16g\n", i, j, Ax [p]) ;
            }
            else if (A->xtype == CHOLMOD_COMPLEX)
            {
                fprintf (ff, ID " " ID " %30.16g %30.16g\n", i, j,
                    Ax [2*p], Ax [2*p+1]) ;
            }
            else // zomplex
            {
                fprintf (ff, ID " " ID " %30.16g %30.16g\n", i, j,
                    Ax [p], Az [p]) ;
            }
        }
    }
    fclose (ff) ;
}

//------------------------------------------------------------------------------
// Int_dump
//------------------------------------------------------------------------------

void Int_dump (Int *P, Int n, char *filename, cholmod_common *cm)
{
    if (!P) return ;
    FILE *ff = fopen (filename, "w") ;
    if (!ff) return ;
    for (Int k = 0 ; k < n ; k++)
    {
        fprintf (ff, ID "\n", P [k]) ;
    }
    fclose (ff) ;
}

//------------------------------------------------------------------------------
// factor_dump
//------------------------------------------------------------------------------

// dump a sparse factorization matrix in triplet form, as an LL' factorization

void factor_dump (cholmod_factor *L, char *L_filename, char *P_filename,
    cholmod_common *cm)
{
    if (!L) return ;
    Int_dump (L->Perm, L->n, P_filename, cm) ;
    cholmod_factor *L2 = CHOLMOD(copy_factor) (L, cm) ;
    cholmod_sparse *A = CHOLMOD(factor_to_sparse) (L2, cm) ;
    sparse_dump (A, L_filename, cm) ;
    CHOLMOD(free_factor) (&L2, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
}

//------------------------------------------------------------------------------
// dense_dump
//------------------------------------------------------------------------------

// dump a dense matrix in triplet form

void dense_dump (cholmod_dense *X, char *filename, cholmod_common *cm)
{
    if (!X) return ;
    cholmod_sparse *A = CHOLMOD(dense_to_sparse) (X, 1, cm) ;
    sparse_dump (A, filename, cm) ;
    CHOLMOD(free_sparse) (&A, cm) ;
}

