//------------------------------------------------------------------------------
// UMFPACK/Source/umfpack_report_info: print Info array
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    User-callable.  Prints the Info array.  See umfpack.h for
    details.
*/

#include "umf_internal.h"

#define PRINT_INFO(format,x) \
{ \
    if (SCALAR_IS_NAN (x) || (!SCALAR_IS_LTZERO (x))) \
    { \
	PRINTF ((format, x)) ; \
    } \
}

/* RATIO macro uses a double relop, but ignore NaN case: */
#define RATIO(a,b,c) (((b) == 0) ? (c) : (((double) a)/((double) b)))

/* ========================================================================== */
/* === print_ratio ========================================================== */
/* ========================================================================== */

PRIVATE void print_ratio
(
    char *what,
    char *format,
    double estimate,
    double actual
)
{
    if (estimate < 0 && actual < 0)	/* double relop, but ignore Nan case */
    {
	return ;
    }
    PRINTF (("    %-27s", what)) ;
    if (estimate >= 0)			/* double relop, but ignore Nan case */
    {
	PRINTF ((format, estimate)) ;
    }
    else
    {
	PRINTF (("                    -")) ;
    }
    if (actual >= 0)			/* double relop, but ignore Nan case */
    {
	PRINTF ((format, actual)) ;
    }
    else
    {
	PRINTF (("                    -")) ;
    }
    if (estimate >= 0 && actual >= 0)	/* double relop, but ignore Nan case */
    {
	PRINTF ((" %5.0f%%\n", 100 * RATIO (actual, estimate, 1))) ;
    }
    else
    {
	PRINTF (("      -\n")) ;
    }
}

/* ========================================================================== */
/* === UMFPACK_report_info ================================================== */
/* ========================================================================== */

void UMFPACK_report_info
(
    const double Control [UMFPACK_CONTROL],
    const double Info [UMFPACK_INFO]
)
{

    double lnz_est, unz_est, lunz_est, lnz, unz, lunz, tsym, tnum, fnum, tsolve,
	fsolve, ftot, twsym, twnum, twsolve, twtot, n2 ;
    Int n_row, n_col, n_inner, prl, is_sym, strategy ;

    /* ---------------------------------------------------------------------- */
    /* get control settings and status to determine what to print */
    /* ---------------------------------------------------------------------- */

    prl = GET_CONTROL (UMFPACK_PRL, UMFPACK_DEFAULT_PRL) ;

    if (!Info || prl < 2)
    {
	/* no output generated if Info is (double *) NULL */
	/* or if prl is less than 2 */
	return ;
    }

    /* ---------------------------------------------------------------------- */
    /* print umfpack version */
    /* ---------------------------------------------------------------------- */

    PRINTF  (("UMFPACK V%d.%d.%d (%s), Info:\n", UMFPACK_MAIN_VERSION,
	UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION, UMFPACK_DATE)) ;

#ifndef NDEBUG
    PRINTF ((
"**** Debugging enabled (UMFPACK will be exceedingly slow!) *****************\n"
    )) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* print run-time options */
    /* ---------------------------------------------------------------------- */

#ifdef DINT
    PRINTF (("    matrix entry defined as:          double\n")) ;
    PRINTF (("    Int (generic integer) defined as: int32_t\n")) ;
#endif
#ifdef DLONG
    PRINTF (("    matrix entry defined as:          double\n")) ;
    PRINTF (("    Int (generic integer) defined as: SuiteSparse_long\n")) ;
#endif
#ifdef ZINT
    PRINTF (("    matrix entry defined as:          double complex\n")) ;
    PRINTF (("    Int (generic integer) defined as: int32_t\n")) ;
#endif
#ifdef ZLONG
    PRINTF (("    matrix entry defined as:          double complex\n")) ;
    PRINTF (("    Int (generic integer) defined as: SuiteSparse_long\n")) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* print compile-time options */
    /* ---------------------------------------------------------------------- */

    PRINTF (("    BLAS library used: ")) ;

#if defined ( MATLAB_MEX_FILE )
    PRINTF (("MATLAB built-in BLAS.  size of BLAS integer: "ID"\n",
	(Int) (sizeof (SUITESPARSE_BLAS_INT)))) ;
#elif defined ( NBLAS )
    PRINTF (("none.  UMFPACK will be slow.\n")) ;
#else
    PRINTF (("%s.  size of BLAS integer: "ID"\n",
        SuiteSparse_BLAS_library ( ),
	(Int) (sizeof (SUITESPARSE_BLAS_INT)))) ;
#endif

    PRINTF (("    MATLAB:                           ")) ;
#ifdef MATLAB_MEX_FILE
    PRINTF (("yes.\n")) ;
#else
#ifdef MATHWORKS
    PRINTF (("yes.\n")) ;
#else
    PRINTF (("no.\n")) ;
#endif
#endif

    PRINTF (("    CPU timer:                        ")) ;
#ifdef SUITESPARSE_TIMER_ENABLED
    PRINTF (("SuiteSparse_time ( ) routine.\n")) ;
#else
    PRINTF (("none.\n")) ;
#endif

    /* ---------------------------------------------------------------------- */
    /* print n and nz */
    /* ---------------------------------------------------------------------- */

    n_row = (Int) Info [UMFPACK_NROW] ;
    n_col = (Int) Info [UMFPACK_NCOL] ;
    n_inner = MIN (n_row, n_col) ;

    PRINT_INFO ("    number of rows in matrix A:       "ID"\n", n_row) ;
    PRINT_INFO ("    number of columns in matrix A:    "ID"\n", n_col) ;
    PRINT_INFO ("    entries in matrix A:              "ID"\n",
	(Int) Info [UMFPACK_NZ]) ;
    PRINT_INFO ("    memory usage reported in:         "ID"-byte Units\n",
	(Int) Info [UMFPACK_SIZE_OF_UNIT]) ;

    PRINT_INFO ("    size of int32_t:                  "ID" bytes\n",
	(Int) Info [UMFPACK_SIZE_OF_INT]) ;
    PRINT_INFO ("    size of SuiteSparse_long:         "ID" bytes\n",
	(Int) Info [UMFPACK_SIZE_OF_LONG]) ;
    PRINT_INFO ("    size of pointer:                  "ID" bytes\n",
	(Int) Info [UMFPACK_SIZE_OF_POINTER]) ;
    PRINT_INFO ("    size of numerical entry:          "ID" bytes\n",
	(Int) Info [UMFPACK_SIZE_OF_ENTRY]) ;

    /* ---------------------------------------------------------------------- */
    /* symbolic parameters */
    /* ---------------------------------------------------------------------- */

    strategy = Info [UMFPACK_STRATEGY_USED] ;
    if (strategy == UMFPACK_STRATEGY_SYMMETRIC)
    {
	PRINTF (("\n    strategy used:                    symmetric\n")) ;
        if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_AMD)
        {
            PRINTF (("    ordering used:                    amd on A+A'\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_GIVEN)
        {
            PRINTF (("    ordering used:                    user perm.\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_USER)
        {
            PRINTF (("    ordering used:                    user function\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_NONE)
        {
            PRINTF (("    ordering used:                    none\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_METIS)
        {
            PRINTF (("    ordering used:                    metis on A+A'\n")) ;
        }
        else
        {
            PRINTF (("    ordering used:                    not computed\n")) ;
        }
    }
    else
    {
	PRINTF (("\n    strategy used:                    unsymmetric\n")) ;
        if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_AMD)
        {
            PRINTF (("    ordering used:                    colamd on A\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_GIVEN)
        {
            PRINTF (("    ordering used:                    user perm.\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_USER)
        {
            PRINTF (("    ordering used:                    user function\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_NONE)
        {
            PRINTF (("    ordering used:                    none\n")) ;
        }
        else if (Info [UMFPACK_ORDERING_USED] == UMFPACK_ORDERING_METIS)
        {
            PRINTF (("    ordering used:                    metis on A'A\n")) ;
        }
        else
        {
            PRINTF (("    ordering used:                    not computed\n")) ;
        }
    }

    if (Info [UMFPACK_QFIXED] == 1)
    {
	PRINTF (("    modify Q during factorization:    no\n")) ;
    }
    else if (Info [UMFPACK_QFIXED] == 0)
    {
	PRINTF (("    modify Q during factorization:    yes\n")) ;
    }

    if (Info [UMFPACK_DIAG_PREFERRED] == 0)
    {
	PRINTF (("    prefer diagonal pivoting:         no\n")) ;
    }
    else if (Info [UMFPACK_DIAG_PREFERRED] == 1)
    {
	PRINTF (("    prefer diagonal pivoting:         yes\n")) ;
    }

    /* ---------------------------------------------------------------------- */
    /* singleton statistics */
    /* ---------------------------------------------------------------------- */

    PRINT_INFO ("    pivots with zero Markowitz cost:               %0.f\n",
	Info [UMFPACK_COL_SINGLETONS] + Info [UMFPACK_ROW_SINGLETONS]) ;
    PRINT_INFO ("    submatrix S after removing zero-cost pivots:\n"
		"        number of \"dense\" rows:                    %.0f\n",
	Info [UMFPACK_NDENSE_ROW]) ;
    PRINT_INFO ("        number of \"dense\" columns:                 %.0f\n",
	Info [UMFPACK_NDENSE_COL]) ;
    PRINT_INFO ("        number of empty rows:                      %.0f\n",
	Info [UMFPACK_NEMPTY_ROW]) ;
    PRINT_INFO ("        number of empty columns                    %.0f\n",
	Info [UMFPACK_NEMPTY_COL]) ;
    is_sym = Info [UMFPACK_S_SYMMETRIC] ;
    if (is_sym > 0)
    {
	PRINTF (("        submatrix S square and diagonal preserved\n")) ;
    }
    else if (is_sym == 0)
    {
	PRINTF (("        submatrix S not square or diagonal not preserved\n"));
    }

    /* ---------------------------------------------------------------------- */
    /* statistics from amd_aat */
    /* ---------------------------------------------------------------------- */

    n2 = Info [UMFPACK_N2] ;
    if (n2 >= 0)
    {
	PRINTF (("    pattern of square submatrix S:\n")) ;
    }
    PRINT_INFO ("        number rows and columns                    %.0f\n",
	n2) ;
    PRINT_INFO ("        symmetry of nonzero pattern:               %.6f\n",
	Info [UMFPACK_PATTERN_SYMMETRY]) ;
    PRINT_INFO ("        nz in S+S' (excl. diagonal):               %.0f\n",
	Info [UMFPACK_NZ_A_PLUS_AT]) ;
    PRINT_INFO ("        nz on diagonal of matrix S:                %.0f\n",
	Info [UMFPACK_NZDIAG]) ;
    if (Info [UMFPACK_NZDIAG] >= 0 && n2 > 0)
    {
	PRINTF (("        fraction of nz on diagonal:                %.6f\n",
	Info [UMFPACK_NZDIAG] / n2)) ;
    }

    /* ---------------------------------------------------------------------- */
    /* statistics from AMD */
    /* ---------------------------------------------------------------------- */

    if (strategy == UMFPACK_STRATEGY_SYMMETRIC && 
        Info [UMFPACK_ORDERING_USED] != UMFPACK_ORDERING_GIVEN)
    {
	double dmax = Info [UMFPACK_SYMMETRIC_DMAX] ;
	PRINTF (("    ordering statistics, for strict diagonal pivoting:\n")) ;
	PRINT_INFO ("        est. flops for LU factorization:           %.5e\n",
	    Info [UMFPACK_SYMMETRIC_FLOPS]) ;
	PRINT_INFO ("        est. nz in L+U (incl. diagonal):           %.0f\n",
	    Info [UMFPACK_SYMMETRIC_LUNZ]) ;
        if (dmax > 0)
        {
            PRINT_INFO
            ("        est. largest front (# entries):            %.0f\n",
	    dmax*dmax) ;
        }
	PRINT_INFO ("        est. max nz in any column of L:            %.0f\n",
	    dmax) ;
	PRINT_INFO (
	    "        number of \"dense\" rows/columns in S+S':    %.0f\n",
	    Info [UMFPACK_SYMMETRIC_NDENSE]) ;
    }

    /* ---------------------------------------------------------------------- */
    /* symbolic factorization */
    /* ---------------------------------------------------------------------- */

    tsym = Info [UMFPACK_SYMBOLIC_TIME] ;
    twsym = Info [UMFPACK_SYMBOLIC_WALLTIME] ;

    PRINT_INFO ("    symbolic factorization defragmentations:       %.0f\n",
	Info [UMFPACK_SYMBOLIC_DEFRAG]) ;
    PRINT_INFO ("    symbolic memory usage (Units):                 %.0f\n",
	Info [UMFPACK_SYMBOLIC_PEAK_MEMORY]) ;
    PRINT_INFO ("    symbolic memory usage (MBytes):                %.1f\n",
	MBYTES (Info [UMFPACK_SYMBOLIC_PEAK_MEMORY])) ;
    PRINT_INFO ("    Symbolic size (Units):                         %.0f\n",
	Info [UMFPACK_SYMBOLIC_SIZE]) ;
    PRINT_INFO ("    Symbolic size (MBytes):                        %.0f\n",
	MBYTES (Info [UMFPACK_SYMBOLIC_SIZE])) ;
    PRINT_INFO ("    symbolic factorization wallclock time(sec):    %.2f\n",
	twsym) ;

    /* ---------------------------------------------------------------------- */
    /* scaling, from numerical factorization */
    /* ---------------------------------------------------------------------- */

    if (Info [UMFPACK_WAS_SCALED] == UMFPACK_SCALE_NONE)
    {
	PRINTF (("\n    matrix scaled: no\n")) ;
    }
    else if (Info [UMFPACK_WAS_SCALED] == UMFPACK_SCALE_SUM)
    {
	PRINTF (("\n    matrix scaled: yes ")) ;
	PRINTF (("(divided each row by sum of abs values in each row)\n")) ;
	PRINTF (("    minimum sum (abs (rows of A)):              %.5e\n",
	    Info [UMFPACK_RSMIN])) ;
	PRINTF (("    maximum sum (abs (rows of A)):              %.5e\n",
	    Info [UMFPACK_RSMAX])) ;
    }
    else if (Info [UMFPACK_WAS_SCALED] == UMFPACK_SCALE_MAX)
    {
	PRINTF (("\n    matrix scaled: yes ")) ;
	PRINTF (("(divided each row by max abs value in each row)\n")) ;
	PRINTF (("    minimum max (abs (rows of A)):              %.5e\n",
	    Info [UMFPACK_RSMIN])) ;
	PRINTF (("    maximum max (abs (rows of A)):              %.5e\n",
	    Info [UMFPACK_RSMAX])) ;
    }

    /* ---------------------------------------------------------------------- */
    /* estimate/actual in symbolic/numeric factorization */
    /* ---------------------------------------------------------------------- */

    /* double relop, but ignore NaN case: */
    if (Info [UMFPACK_SYMBOLIC_DEFRAG] >= 0	/* UMFPACK_*symbolic called */
    ||  Info [UMFPACK_NUMERIC_DEFRAG] >= 0)	/* UMFPACK_numeric called */
    {
	PRINTF (("\n    symbolic/numeric factorization:      upper bound")) ;
	PRINTF (("               actual      %%\n")) ;
	PRINTF (("    variable-sized part of Numeric object:\n")) ;
    }
    print_ratio ("    initial size (Units)", " %20.0f",
	Info [UMFPACK_VARIABLE_INIT_ESTIMATE], Info [UMFPACK_VARIABLE_INIT]) ;
    print_ratio ("    peak size (Units)", " %20.0f",
	Info [UMFPACK_VARIABLE_PEAK_ESTIMATE], Info [UMFPACK_VARIABLE_PEAK]) ;
    print_ratio ("    final size (Units)", " %20.0f",
	Info [UMFPACK_VARIABLE_FINAL_ESTIMATE], Info [UMFPACK_VARIABLE_FINAL]) ;
    print_ratio ("Numeric final size (Units)", " %20.0f",
	Info [UMFPACK_NUMERIC_SIZE_ESTIMATE], Info [UMFPACK_NUMERIC_SIZE]) ;
    print_ratio ("Numeric final size (MBytes)", " %20.1f",
	MBYTES (Info [UMFPACK_NUMERIC_SIZE_ESTIMATE]),
	MBYTES (Info [UMFPACK_NUMERIC_SIZE])) ;
    print_ratio ("peak memory usage (Units)", " %20.0f",
	Info [UMFPACK_PEAK_MEMORY_ESTIMATE], Info [UMFPACK_PEAK_MEMORY]) ;
    print_ratio ("peak memory usage (MBytes)", " %20.1f",
	MBYTES (Info [UMFPACK_PEAK_MEMORY_ESTIMATE]),
	MBYTES (Info [UMFPACK_PEAK_MEMORY])) ;
    print_ratio ("numeric factorization flops", " %20.5e",
	Info [UMFPACK_FLOPS_ESTIMATE], Info [UMFPACK_FLOPS]) ;

    lnz_est = Info [UMFPACK_LNZ_ESTIMATE] ;
    unz_est = Info [UMFPACK_UNZ_ESTIMATE] ;
    if (lnz_est >= 0 && unz_est >= 0)	/* double relop, but ignore NaN case */
    {
	lunz_est = lnz_est + unz_est - n_inner ;
    }
    else
    {
	lunz_est = EMPTY ;
    }
    lnz = Info [UMFPACK_LNZ] ;
    unz = Info [UMFPACK_UNZ] ;
    if (lnz >= 0 && unz >= 0)		/* double relop, but ignore NaN case */
    {
	lunz = lnz + unz - n_inner ;
    }
    else
    {
	lunz = EMPTY ;
    }
    print_ratio ("nz in L (incl diagonal)", " %20.0f", lnz_est, lnz) ;
    print_ratio ("nz in U (incl diagonal)", " %20.0f", unz_est, unz) ;
    print_ratio ("nz in L+U (incl diagonal)", " %20.0f", lunz_est, lunz) ;

    print_ratio ("largest front (# entries)", " %20.0f",
	Info [UMFPACK_MAX_FRONT_SIZE_ESTIMATE], Info [UMFPACK_MAX_FRONT_SIZE]) ;
    print_ratio ("largest # rows in front", " %20.0f",
	Info [UMFPACK_MAX_FRONT_NROWS_ESTIMATE],
	Info [UMFPACK_MAX_FRONT_NROWS]) ;
    print_ratio ("largest # columns in front", " %20.0f",
	Info [UMFPACK_MAX_FRONT_NCOLS_ESTIMATE],
	Info [UMFPACK_MAX_FRONT_NCOLS]) ;

    /* ---------------------------------------------------------------------- */
    /* numeric factorization */
    /* ---------------------------------------------------------------------- */

    tnum = Info [UMFPACK_NUMERIC_TIME] ;
    twnum = Info [UMFPACK_NUMERIC_WALLTIME] ;
    fnum = Info [UMFPACK_FLOPS] ;

    PRINT_INFO ("\n    initial allocation ratio used:                 %0.3g\n",
	Info [UMFPACK_ALLOC_INIT_USED]) ;
    PRINT_INFO ("    # of forced updates due to frontal growth:     %.0f\n",
	Info [UMFPACK_FORCED_UPDATES]) ;
    PRINT_INFO ("    number of off-diagonal pivots:                 %.0f\n",
	Info [UMFPACK_NOFF_DIAG]) ;
    PRINT_INFO ("    nz in L (incl diagonal), if none dropped       %.0f\n",
	Info [UMFPACK_ALL_LNZ]) ;
    PRINT_INFO ("    nz in U (incl diagonal), if none dropped       %.0f\n",
	Info [UMFPACK_ALL_UNZ]) ;
    PRINT_INFO ("    number of small entries dropped                %.0f\n",
	Info [UMFPACK_NZDROPPED]) ;
    PRINT_INFO ("    nonzeros on diagonal of U:                     %.0f\n",
	Info [UMFPACK_UDIAG_NZ]) ;
    PRINT_INFO ("    min abs. value on diagonal of U:               %.2e\n",
	Info [UMFPACK_UMIN]) ;
    PRINT_INFO ("    max abs. value on diagonal of U:               %.2e\n",
	Info [UMFPACK_UMAX]) ;
    PRINT_INFO ("    estimate of reciprocal of condition number:    %.2e\n",
	Info [UMFPACK_RCOND]) ;
    PRINT_INFO ("    indices in compressed pattern:                 %.0f\n",
	Info [UMFPACK_COMPRESSED_PATTERN]) ;
    PRINT_INFO ("    numerical values stored in Numeric object:     %.0f\n",
	Info [UMFPACK_LU_ENTRIES]) ;
    PRINT_INFO ("    numeric factorization defragmentations:        %.0f\n",
	Info [UMFPACK_NUMERIC_DEFRAG]) ;
    PRINT_INFO ("    numeric factorization reallocations:           %.0f\n",
	Info [UMFPACK_NUMERIC_REALLOC]) ;
    PRINT_INFO ("    costly numeric factorization reallocations:    %.0f\n",
	Info [UMFPACK_NUMERIC_COSTLY_REALLOC]) ;
    PRINT_INFO ("    numeric factorization wallclock time (sec):    %.2f\n",
	twnum) ;

#define TMIN 0.001

    if (twnum > TMIN && fnum > 0)
    {
	PRINT_INFO (
	   "    numeric factorization mflops (wallclock):      %.2f\n",
	   1e-6 * fnum / twnum) ;
    }

    ftot = fnum ;

    twtot = EMPTY ;
    if (twsym >= TMIN && twnum >= TMIN)
    {
	twtot = twsym + twnum ;
	PRINT_INFO ("    symbolic + numeric wall clock time (sec):      %.2f\n",
	    twtot) ;
	if (ftot > 0 && twtot > TMIN)
	{
	    PRINT_INFO (
		"    symbolic + numeric mflops (wall clock):        %.2f\n",
		1e-6 * ftot / twtot) ;
	}
    }

    /* ---------------------------------------------------------------------- */
    /* solve */
    /* ---------------------------------------------------------------------- */

    tsolve = Info [UMFPACK_SOLVE_TIME] ;
    twsolve = Info [UMFPACK_SOLVE_WALLTIME] ;
    fsolve = Info [UMFPACK_SOLVE_FLOPS] ;

    PRINT_INFO ("\n    solve flops:                                   %.5e\n",
	fsolve) ;
    PRINT_INFO ("    iterative refinement steps taken:              %.0f\n",
	Info [UMFPACK_IR_TAKEN]) ;
    PRINT_INFO ("    iterative refinement steps attempted:          %.0f\n",
	Info [UMFPACK_IR_ATTEMPTED]) ;
    PRINT_INFO ("    sparse backward error omega1:                  %.2e\n",
	Info [UMFPACK_OMEGA1]) ;
    PRINT_INFO ("    sparse backward error omega2:                  %.2e\n",
	Info [UMFPACK_OMEGA2]) ;
    PRINT_INFO ("    solve wall clock time (sec):                   %.2f\n",
	twsolve) ;
    if (fsolve > 0 && twsolve > TMIN)
    {
	PRINT_INFO (
	    "    solve mflops (wall clock time):                %.2f\n",
	    1e-6 * fsolve / twsolve) ;
    }

    if (ftot >= 0 && fsolve >= 0)
    {
	ftot += fsolve ;
	PRINT_INFO (
	"\n    total symbolic + numeric + solve flops:        %.5e\n", ftot) ;
    }

    if (twsolve >= TMIN)
    {
	if (twtot >= TMIN && ftot >= 0)
	{
	    twtot += tsolve ;
	    PRINT_INFO (
		"    total symbolic+numeric+solve wall clock time:  %.2f\n",
		twtot) ;
	    if (ftot > 0 && twtot > TMIN)
	    {
		PRINT_INFO (
		"    total symbolic+numeric+solve mflops(wallclock) %.2f\n",
		1e-6 * ftot / twtot) ;
	    }
	}
    }
    PRINTF (("\n")) ;
}
