//------------------------------------------------------------------------------
// SuiteSparse/KLU/User/klu_cholmod.h: include file for klu_cholmod library
//------------------------------------------------------------------------------

// KLU, Copyright (c) 2004-2022, University of Florida.  All Rights Reserved.
// Authors: Timothy A. Davis and Ekanathan Palamadai.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "klu.h"

int32_t klu_cholmod (int32_t n, int32_t Ap [ ],
    int32_t Ai [ ], int32_t Perm [ ], klu_common *) ;

SuiteSparse_long klu_l_cholmod (SuiteSparse_long n, SuiteSparse_long Ap [ ],
    SuiteSparse_long Ai [ ], SuiteSparse_long Perm [ ], klu_l_common *) ;

