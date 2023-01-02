//------------------------------------------------------------------------------
// UMFPACK/Demo/umf4_f77wrapper_methods: Fortran interface for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

// This template file is #include'd into umf4_f77wrapper.c to create different
// sets of methods with different Fortran names.

/* -------------------------------------------------------------------------- */
/* umf4def: set default control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4def (control) */

void umf4def_FORTRAN (double Control [UMFPACK_CONTROL])
{
    UMFPACK_defaults (Control) ;
}

/* -------------------------------------------------------------------------- */
/* umf4pcon: print control parameters */
/* -------------------------------------------------------------------------- */

/* call umf4pcon (control) */

void umf4pcon_FORTRAN (double Control [UMFPACK_CONTROL])
{
    fflush (stdout) ;
    UMFPACK_report_control (Control) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4sym: pre-ordering and symbolic factorization */
/* -------------------------------------------------------------------------- */

/* call umf4sym (m, n, Ap, Ai, Ax, symbolic, control, info) */

void umf4sym_FORTRAN (Int *m, Int *n, Int Ap [ ], Int Ai [ ],
    double Ax [ ], void **Symbolic,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_symbolic (*m, *n, Ap, Ai, Ax, Symbolic, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4num: numeric factorization */
/* -------------------------------------------------------------------------- */

/* call umf4num (Ap, Ai, Ax, symbolic, numeric, control, info) */

void umf4num_FORTRAN (Int Ap [ ], Int Ai [ ], double Ax [ ],
    void **Symbolic, void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_numeric (Ap, Ai, Ax, *Symbolic, Numeric, Control, Info);
}

/* -------------------------------------------------------------------------- */
/* umf4solr: solve a linear system with iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4solr (sys, Ap, Ai, Ax, x, b, numeric, control, info) */

void umf4solr_FORTRAN (Int *sys, Int Ap [ ], Int Ai [ ], double Ax [ ],
    double x [ ], double b [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    (void) UMFPACK_solve (*sys, Ap, Ai, Ax, x, b, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4sol: solve a linear system without iterative refinement */
/* -------------------------------------------------------------------------- */

/* call umf4sol (sys, x, b, numeric, control, info) */

void umf4sol_FORTRAN (Int *sys, double x [ ], double b [ ], void **Numeric,
    double Control [UMFPACK_CONTROL], double Info [UMFPACK_INFO])
{
    Control [UMFPACK_IRSTEP] = 0 ;
    (void) UMFPACK_solve (*sys, (Int *) NULL, (Int *) NULL, (double *) NULL,
	x, b, *Numeric, Control, Info) ;
}

/* -------------------------------------------------------------------------- */
/* umf4scal: scale a vector using UMFPACK's scale factors */
/* -------------------------------------------------------------------------- */

/* call umf4scal (x, b, numeric, status) */

void umf4scal_FORTRAN (double x [ ], double b [ ], void **Numeric, Int *status)
{
    *status = UMFPACK_scale (x, b, *Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4pinf: print info */
/* -------------------------------------------------------------------------- */

/* call umf4pinf (control) */

void umf4pinf_FORTRAN (double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO])
{
    fflush (stdout) ;
    UMFPACK_report_info (Control, Info) ;
    fflush (stdout) ;
}

/* -------------------------------------------------------------------------- */
/* umf4fnum: free the Numeric object */
/* -------------------------------------------------------------------------- */

/* call umf4fnum (numeric) */

void umf4fnum_FORTRAN (void **Numeric)
{
    UMFPACK_free_numeric (Numeric) ;
}

/* -------------------------------------------------------------------------- */
/* umf4fsym: free the Symbolic object */
/* -------------------------------------------------------------------------- */

/* call umf4fsym (symbolic) */

void umf4fsym_FORTRAN (void **Symbolic)
{
    UMFPACK_free_symbolic (Symbolic) ;
}

/* -------------------------------------------------------------------------- */
/* umf4snum: save the Numeric object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4snum (numeric, filenum, status) */

void umf4snum_FORTRAN (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_save_numeric (*Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4ssym: save the Symbolic object to a file */
/* -------------------------------------------------------------------------- */

/* call umf4ssym (symbolic, filenum, status) */

void umf4ssym_FORTRAN (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_save_symbolic (*Symbolic, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4lnum: load the Numeric object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4lnum (numeric, filenum, status) */

void umf4lnum_FORTRAN (void **Numeric, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "n", filename) ;
    *status = UMFPACK_load_numeric (Numeric, filename) ;
}

/* -------------------------------------------------------------------------- */
/* umf4lsym: load the Symbolic object from a file */
/* -------------------------------------------------------------------------- */

/* call umf4lsym (symbolic, filenum, status) */

void umf4lsym_FORTRAN (void **Symbolic, Int *filenum, Int *status)
{
    char filename [LEN] ;
    make_filename (*filenum, "s", filename) ;
    *status = UMFPACK_load_symbolic (Symbolic, filename) ;
}

#undef umf4def_FORTRAN
#undef umf4pcon_FORTRAN
#undef umf4sym_FORTRAN
#undef umf4num_FORTRAN
#undef umf4solr_FORTRAN
#undef umf4sol_FORTRAN
#undef umf4scal_FORTRAN
#undef umf4pinf_FORTRAN
#undef umf4fnum_FORTRAN
#undef umf4fsym_FORTRAN
#undef umf4snum_FORTRAN
#undef umf4ssym_FORTRAN
#undef umf4lnum_FORTRAN
#undef umf4lsym_FORTRAN

