//------------------------------------------------------------------------------
// UMFPACK/Source/umf_cholmod.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

#include "umfpack.h"

int umf_i_cholmod
(
    /* inputs */
    int32_t nrow,           /* A is nrow-by-ncol */
    int32_t ncol,           /* A is nrow-by-ncol */
    int32_t symmetric,      /* if true and nrow=ncol do A+A', else do A'A */
    int32_t Ap [ ],         /* column pointers, size ncol+1 */
    int32_t Ai [ ],         /* row indices, size nz = Ap [ncol] */
    /* output */
    int32_t Perm [ ],       /* fill-reducing permutation, size ncol */
    /* user-defined */
    void *ignore,           /* not needed */
    double user_info [3]    /* [0]: max col count for L=chol(P(A+A')P')
                               [1]: nnz (L)
                               [2]: flop count for chol, if A real */
) ;

int umf_l_cholmod
(
    /* inputs */
    int64_t nrow,           /* A is nrow-by-ncol */
    int64_t ncol,           /* A is nrow-by-ncol */
    int64_t symmetric,      /* if true and nrow=ncol do A+A', else do A'A */
    int64_t Ap [ ],         /* column pointers, size ncol+1 */
    int64_t Ai [ ],         /* row indices, size nz = Ap [ncol] */
    /* output */
    int64_t Perm [ ],       /* fill-reducing permutation, size ncol */
    /* user-defined */
    void *ignore,           /* not needed */
    double user_info [3]    /* [0]: max col count for L=chol(P(A+A')P')
                               [1]: nnz (L)
                               [2]: flop count for chol, if A real */
) ;

