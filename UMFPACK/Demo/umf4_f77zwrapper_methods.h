//------------------------------------------------------------------------------
// UMFPACK/Demo/umf4_f77zwrapper_methods: Fortran interface for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// This template file is #include'd into umf4_f77zwrapper.c to create different
// sets of methods with different Fortran names.

/* -------------------------------------------------------------------------- */
/* umf4zdef: set default control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4zdef (control) */

void umf4zdef_FORTRAN (double Control [UMFPACK_CONTROL])
{
    UMFPACK_defaults (Control) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zpcon: print control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4zpcon (control) */

void umf4zpcon_FORTRAN (double Control [UMFPACK_CONTROL])
{
    fflush (stdout) ;
    UMFPACK_report_control (Control) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsym: pre-ordering and symbolic factorization */
/* -------------------------------------------------------------------------- */

/* call umf4zsym (m, n, Ap, Ai, Ax, Az, symbolic, control, info) */

void umf4zsym_FORTRAN (Int *m, Int *n, Int Ap [ ], Int Ai [ ],
    double Ax [ ], double Az [ ], void **Symbolic,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_symbolic (*m, *n, Ap, Ai, Ax, Az, Symbolic, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4znum: numeric factorization */
/* -------------------------------------------------------------------------- */

/* call umf4znum (Ap, Ai, Ax, Az, symbolic, numeric, control, info) */

void umf4znum_FORTRAN (Int Ap [ ], Int Ai [ ], double Ax [ ], double Az [ ],
    void **Symbolic, void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_numeric (Ap, Ai, Ax, Az, *Symbolic, Numeric, Control, Info);
}

/* -------------------------------------------------------------------------- */
/* umf4zsolr: solve a linear system with iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4zsolr (sys, Ap, Ai, Ax, Az, x, xz, b, bz, numeric, control, info) */

void umf4zsolr_FORTRAN (Int *sys, Int Ap [ ], Int Ai [ ],
    double Ax [ ], double Az [ ],
    double x [ ], double xz [ ], double b [ ], double bz [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_solve (*sys, Ap, Ai, Ax, Az, x, xz, b, bz,
	*Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsol: solve a linear system without iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4zsol (sys, x, xz, b, bz, numeric, control, info) */

void umf4zsol_FORTRAN (Int *sys, double x [ ], double xz [ ], double b [ ],
    double bz [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    Control [UMFPACK_IRSTEP] = 0 ;
    (void) UMFPACK_solve (*sys, (Int *) NULL, (Int *) NULL, (double *) NULL,
	(double *) NULL, x, xz, b, bz, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zscal: scale a vector using UMFPACK's scale factors */
/* -------------------------------------------------------------------------- */

/* call umf4zscal (x, xz, b, bz, numeric, status) */

void umf4zscal_FORTRAN (double x [ ], double xz [ ],
    double b [ ], double bz [ ],
    void **Numeric, Int *status)
{
    *status = UMFPACK_scale (x, xz, b, bz, *Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zpinf: print info */
/* -------------------------------------------------------------------------- */

/* call umf4zpinf (control) */

void umf4zpinf_FORTRAN (double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO])
{
    fflush (stdout) ;
    UMFPACK_report_info (Control, Info) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zfnum: free the Numeric object */
/* -------------------------------------------------------------------------- */

/* call umf4zfnum (numeric) */

void umf4zfnum_FORTRAN (void **Numeric)
{
    UMFPACK_free_numeric (Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zfsym: free the Symbolic object */
/* -------------------------------------------------------------------------- */

/* call umf4zfsym (symbolic) */

void umf4zfsym_FORTRAN (void **Symbolic)
{
    UMFPACK_free_symbolic (Symbolic) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zsnum: save the Numeric object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4zsnum (numeric, filenum, status) */

void umf4zsnum_FORTRAN (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_save_numeric (*Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zssym: save the Symbolic object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4zssym (symbolic, filenum, status) */

void umf4zssym_FORTRAN (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_save_symbolic (*Symbolic, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zlnum: load the Numeric object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4zlnum (numeric, filenum, status) */

void umf4zlnum_FORTRAN (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_load_numeric (Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4zlsym: load the Symbolic object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4zlsym (symbolic, filenum, status) */

void umf4zlsym_FORTRAN (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_load_symbolic (Symbolic, filename) ;
}

#undef umf4zdef_FORTRAN
#undef umf4zpcon_FORTRAN
#undef umf4zsym_FORTRAN
#undef umf4znum_FORTRAN
#undef umf4zsolr_FORTRAN
#undef umf4zsol_FORTRAN
#undef umf4zscal_FORTRAN
#undef umf4zpinf_FORTRAN
#undef umf4zfnum_FORTRAN
#undef umf4zfsym_FORTRAN
#undef umf4zsnum_FORTRAN
#undef umf4zssym_FORTRAN
#undef umf4zlnum_FORTRAN
#undef umf4zlsym_FORTRAN

