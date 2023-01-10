//------------------------------------------------------------------------------
// UMFPACK/Source/umf_transpose.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

Int UMF_transpose
(
    Int n_row,
    Int n_col,
    const Int Ap [ ],
    const Int Ai [ ],
    const double Ax [ ],
    const Int P [ ],
    const Int Q [ ],
    Int nq,
    Int Rp [ ],
    Int Ri [ ],
    double Rx [ ],
    Int W [ ],
    Int check
#ifdef COMPLEX
    , const double Az [ ]
    , double Rz [ ]
    , Int do_conjugate
#endif
) ;
