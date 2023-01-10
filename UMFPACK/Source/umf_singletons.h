//------------------------------------------------------------------------------
// UMFPACK/Source/umf_singletons.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

Int UMF_singletons
(
    Int n_row,
    Int n_col,
    const Int Ap [ ],
    const Int Ai [ ],
    const Int Quser [ ],
    Int strategy,
    Int do_singletons,
    Int Cdeg [ ],
    Int Cperm [ ],
    Int Rdeg [ ],
    Int Rperm [ ],
    Int InvRperm [ ],
    Int *n1,
    Int *n1c,
    Int *n1r,
    Int *nempty_col,
    Int *nempty_row,
    Int *is_sym,
    Int *max_rdeg,
    Int Rp [ ],
    Int Ri [ ],
    Int W [ ],
    Int Next [ ]
) ;
