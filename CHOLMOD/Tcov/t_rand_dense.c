//------------------------------------------------------------------------------
// CHOLMOD/Tcov/t_rand_dense: random dense matrix
//------------------------------------------------------------------------------

// CHOLMOD/Tcov Module.  Copyright (C) 2005-2023, Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

cholmod_dense *rand_dense
(
    Int nrow,
    Int ncol,
    int xdtype,
    cholmod_common *Common
)
{

    cholmod_dense *X = CHOLMOD(zeros) (nrow, ncol, xdtype, Common) ;
    if (!X) return (NULL) ;
    Int nz = nrow*ncol ;

    switch (xdtype % 8)
    {
        case CHOLMOD_REAL    + CHOLMOD_SINGLE:
            {
                float *Xx = X->x ;
                for (Int p = 0 ; p < nz ; p++)
                {
                    Xx [p] = xrand (1) ;                // RAND
                }
            }
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_SINGLE:
            {
                float *Xx = X->x ;
                for (Int p = 0 ; p < 2*nz ; p++)
                {
                    Xx [p] = xrand (1) ;                // RAND
                }
            }
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_SINGLE:
            {
                float *Xx = X->x ;
                float *Xz = X->z ;
                for (Int p = 0 ; p < nz ; p++)
                {
                    Xx [p] = xrand (1) ;                // RAND
                    Xz [p] = xrand (1) ;                // RAND
                }
            }
            break ;

        case CHOLMOD_REAL    + CHOLMOD_DOUBLE:
            {
                double *Xx = X->x ;
                for (Int p = 0 ; p < nz ; p++)
                {
                    Xx [p] = xrand (1) ;                // RAND
                }
            }
            break ;

        case CHOLMOD_COMPLEX + CHOLMOD_DOUBLE:
            {
                double *Xx = X->x ;
                for (Int p = 0 ; p < 2*nz ; p++)
                {
                    Xx [p] = xrand (1) ;                // RAND
                }
            }
            break ;

        case CHOLMOD_ZOMPLEX + CHOLMOD_DOUBLE:
            {
                double *Xx = X->x ;
                double *Xz = X->z ;
                for (Int p = 0 ; p < nz ; p++)
                {
                    Xx [p] = xrand (1) ;                // RAND
                    Xz [p] = xrand (1) ;                // RAND
                }
            }
            break ;
    }

    return (X) ;
}

