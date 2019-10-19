/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License for License.                      */
/* -------------------------------------------------------------------------- */

GLOBAL Int UMF_analyze
(
    Int n_row,
    Int n_col,
    Int Ai [ ],
    Int Ap [ ],
    Int Up [ ],
    Int fixQ,
    Int Front_ncols [ ],
    Int W [ ],
    Int Link [ ],
    Int Front_nrows [ ],
    Int Front_npivcol [ ],
    Int Front_parent [ ],
    Int *nfr_out,
    Int *p_ncompactions
) ;
