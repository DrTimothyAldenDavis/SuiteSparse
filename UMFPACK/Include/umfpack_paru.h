/* ========================================================================== */
/* === umfpack_paru ========================================================== */
/* ========================================================================== */

/* -------------------------------------------------------------------------- */
/* Copyright (c) 2005-2012 by Timothy A. Davis, http://www.suitesparse.com.   */
/* All Rights Reserved.  See ../Doc/License.txt for License.                  */
/* -------------------------------------------------------------------------- */

int umfpack_di_paru_symbolic
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ],
    const double Ax [ ],
    const int Qinit [ ],
    int (*user_ordering) ( int, int, int, int *, int *, int *, void *,
        double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

SuiteSparse_long umfpack_dl_paru_symbolic
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    const double Ax [ ],
    const SuiteSparse_long Qinit [ ],
    int (*user_ordering) (SuiteSparse_long, SuiteSparse_long, SuiteSparse_long,
        SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *, void *,
        double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_paru_symbolic
(
    int n_row,
    int n_col,
    const int Ap [ ],
    const int Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int Qinit [ ],
    int (*user_ordering) (int, int, int, int *, int *, int *, void *, double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

SuiteSparse_long umfpack_zl_paru_symbolic
(
    SuiteSparse_long n_row,
    SuiteSparse_long n_col,
    const SuiteSparse_long Ap [ ],
    const SuiteSparse_long Ai [ ],
    const double Ax [ ], const double Az [ ],
    const SuiteSparse_long Qinit [ ],
    int (*user_ordering) (SuiteSparse_long, SuiteSparse_long, SuiteSparse_long,
        SuiteSparse_long *, SuiteSparse_long *, SuiteSparse_long *, void *,
        double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

void umfpack_di_paru_free_sw
(
    void **SW
) ;

void umfpack_dl_paru_free_sw
(
    void **SW
) ;

void umfpack_zi_paru_free_sw
(
    void **SW
) ;

void umfpack_zl_paru_free_sw
(
    void **SW
) ;

