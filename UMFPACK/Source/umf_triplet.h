//------------------------------------------------------------------------------
// UMFPACK/Source/umf_triplet.h
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

Int UMF_triplet_map_x
(
    Int n_row,
    Int n_col,
    Int nz,
    const Int Ti [ ],
    const Int Tj [ ],
    Int Ap [ ],
    Int Ai [ ],
    Int Rp [ ],
    Int Rj [ ],
    Int W [ ],
    Int RowCount [ ]
    , const double Tx [ ]
    , double Ax [ ]
    , double Rx [ ]
#ifdef COMPLEX
    , const double Tz [ ]
    , double Az [ ]
    , double Rz [ ]
#endif
    , Int Map [ ]
    , Int Map2 [ ]
) ;

Int UMF_triplet_map_nox
(
    Int n_row,
    Int n_col,
    Int nz,
    const Int Ti [ ],
    const Int Tj [ ],
    Int Ap [ ],
    Int Ai [ ],
    Int Rp [ ],
    Int Rj [ ],
    Int W [ ],
    Int RowCount [ ]
    , Int Map [ ]
    , Int Map2 [ ]
) ;

Int UMF_triplet_nomap_x
(
    Int n_row,
    Int n_col,
    Int nz,
    const Int Ti [ ],
    const Int Tj [ ],
    Int Ap [ ],
    Int Ai [ ],
    Int Rp [ ],
    Int Rj [ ],
    Int W [ ],
    Int RowCount [ ]
    , const double Tx [ ]
    , double Ax [ ]
    , double Rx [ ]
#ifdef COMPLEX
    , const double Tz [ ]
    , double Az [ ]
    , double Rz [ ]
#endif
) ;

Int UMF_triplet_nomap_nox
(
    Int n_row,
    Int n_col,
    Int nz,
    const Int Ti [ ],
    const Int Tj [ ],
    Int Ap [ ],
    Int Ai [ ],
    Int Rp [ ],
    Int Rj [ ],
    Int W [ ],
    Int RowCount [ ]
) ;
