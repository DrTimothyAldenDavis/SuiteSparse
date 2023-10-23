//------------------------------------------------------------------------------
// UMFPACK/Include/umfpack.h: include file for UMFPACK
//------------------------------------------------------------------------------

// UMFPACK, Copyright (c) 2005-2023, Timothy A. Davis, All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0+

//------------------------------------------------------------------------------

/*
    This is the umfpack.h include file, and should be included in all user code
    that uses UMFPACK.  Do not include any of the umf_* header files in user
    code.  All routines in UMFPACK starting with "umfpack_" are user-callable.
    All other routines are prefixed "umf_XY_", (where X is d or z, and Y is
    i or l) and are not user-callable.
*/

#ifndef UMFPACK_H
#define UMFPACK_H

/* -------------------------------------------------------------------------- */
/* Make it easy for C++ programs to include UMFPACK */
/* -------------------------------------------------------------------------- */

#ifdef __cplusplus
extern "C" {
#endif

//------------------------------------------------------------------------------
// include files for other packages: SuiteSparse_config and AMD
//------------------------------------------------------------------------------

#include "SuiteSparse_config.h"
#include "amd.h"

/* -------------------------------------------------------------------------- */
/* size of Info and Control arrays */
/* -------------------------------------------------------------------------- */

/* These might be larger in future versions, since there are only 3 unused
 * entries in Info, and no unused entries in Control. */

#define UMFPACK_INFO 90
#define UMFPACK_CONTROL 20

/* -------------------------------------------------------------------------- */
/* Version, copyright, and license */
/* -------------------------------------------------------------------------- */

#define UMFPACK_COPYRIGHT \
"UMFPACK:  Copyright (c) 2005-2023 by Timothy A. Davis.  All Rights Reserved.\n"

#define UMFPACK_LICENSE_PART1 \
"\nUMFPACK License: SPDX-License-Identifier: GPL-2.0+\n" \
"   UMFPACK is available under alternate licenses,\n" \
"   contact T. Davis for details.\n" 

#define UMFPACK_LICENSE_PART2 "\n"

#define UMFPACK_LICENSE_PART3 \
"\n" \
"Availability: http://www.suitesparse.com" \
"\n"

/* UMFPACK Version 4.5 and later will include the following definitions.
 * As an example, to test if the version you are using is 4.5 or later:
 *
 * #ifdef UMFPACK_VER
 *      if (UMFPACK_VER >= UMFPACK_VER_CODE (4,5)) ...
 * #endif
 *
 * This also works during compile-time:
 *
 *      #if defined(UMFPACK_VER) && (UMFPACK >= UMFPACK_VER_CODE (4,5))
 *          printf ("This is version 4.5 or later\n") ;
 *      #else
 *          printf ("This is an early version\n") ;
 *      #endif
 *
 * Versions 4.4 and earlier of UMFPACK do not include a #define'd version
 * number, although they do include the UMFPACK_VERSION string, defined
 * below.
 */

#define UMFPACK_DATE "Oct 23, 2023"
#define UMFPACK_MAIN_VERSION   6
#define UMFPACK_SUB_VERSION    2
#define UMFPACK_SUBSUB_VERSION 2

#define UMFPACK_VER_CODE(main,sub) ((main) * 1000 + (sub))
#define UMFPACK_VER UMFPACK_VER_CODE(UMFPACK_MAIN_VERSION,UMFPACK_SUB_VERSION)

// user code should not directly use GB_STR or GB_XSTR
// GB_STR: convert the content of x into a string "x"
#define GB_XSTR(x) GB_STR(x)
#define GB_STR(x) #x

#define UMFPACK_VERSION "UMFPACK V"                                 \
    GB_XSTR(UMFPACK_MAIN_VERSION) "."                               \
    GB_XSTR(UMFPACK_SUB_VERSION) "."                                \
    GB_XSTR(UMFPACK_SUBSUB_VERSION) " (" UMFPACK_DATE ")"

/* -------------------------------------------------------------------------- */
/* contents of Info */
/* -------------------------------------------------------------------------- */

/* Note that umfpack_report.m must coincide with these definitions.  S is
 * the submatrix of A after removing row/col singletons and empty rows/cols. */

/* returned by all routines that use Info: */
#define UMFPACK_STATUS 0        /* UMFPACK_OK, or other result */
#define UMFPACK_NROW 1          /* n_row input value */
#define UMFPACK_NCOL 16         /* n_col input value */
#define UMFPACK_NZ 2            /* # of entries in A */

/* computed in UMFPACK_*symbolic and UMFPACK_numeric: */
#define UMFPACK_SIZE_OF_UNIT 3          /* sizeof (Unit) */

/* computed in UMFPACK_*symbolic: */
#define UMFPACK_SIZE_OF_INT 4           /* sizeof (int32_t) */
#define UMFPACK_SIZE_OF_LONG 5          /* sizeof (int64_t) */
#define UMFPACK_SIZE_OF_POINTER 6       /* sizeof (void *) */
#define UMFPACK_SIZE_OF_ENTRY 7         /* sizeof (Entry), real or complex */
#define UMFPACK_NDENSE_ROW 8            /* number of dense rows */
#define UMFPACK_NEMPTY_ROW 9            /* number of empty rows */
#define UMFPACK_NDENSE_COL 10           /* number of dense rows */
#define UMFPACK_NEMPTY_COL 11           /* number of empty rows */
#define UMFPACK_SYMBOLIC_DEFRAG 12      /* # of memory compactions */
#define UMFPACK_SYMBOLIC_PEAK_MEMORY 13 /* memory used by symbolic analysis */
#define UMFPACK_SYMBOLIC_SIZE 14        /* size of Symbolic object, in Units */
#define UMFPACK_SYMBOLIC_TIME 15        /* time (sec.) for symbolic analysis */
#define UMFPACK_SYMBOLIC_WALLTIME 17    /* wall clock time for sym. analysis */
#define UMFPACK_STRATEGY_USED 18        /* strategy used: sym, unsym */
#define UMFPACK_ORDERING_USED 19        /* ordering used: colamd, amd, given */
#define UMFPACK_QFIXED 31               /* whether Q is fixed or refined */
#define UMFPACK_DIAG_PREFERRED 32       /* whether diagonal pivoting attempted*/
#define UMFPACK_PATTERN_SYMMETRY 33     /* symmetry of pattern of S */
#define UMFPACK_NZ_A_PLUS_AT 34         /* nnz (S+S'), excl. diagonal */
#define UMFPACK_NZDIAG 35               /* nnz (diag (S)) */

/* AMD statistics, computed in UMFPACK_*symbolic: */
#define UMFPACK_SYMMETRIC_LUNZ 36       /* nz in L+U, if AMD ordering used */
#define UMFPACK_SYMMETRIC_FLOPS 37      /* flops for LU, if AMD ordering used */
#define UMFPACK_SYMMETRIC_NDENSE 38     /* # of "dense" rows/cols in S+S' */
#define UMFPACK_SYMMETRIC_DMAX 39       /* max nz in cols of L, for AMD */

/* 51:55 unused */

/* statistcs for singleton pruning */
#define UMFPACK_COL_SINGLETONS 56       /* # of column singletons */
#define UMFPACK_ROW_SINGLETONS 57       /* # of row singletons */
#define UMFPACK_N2 58                   /* size of S */
#define UMFPACK_S_SYMMETRIC 59          /* 1 if S square and symmetricly perm.*/

/* estimates computed in UMFPACK_*symbolic: */
#define UMFPACK_NUMERIC_SIZE_ESTIMATE 20    /* final size of Numeric->Memory */
#define UMFPACK_PEAK_MEMORY_ESTIMATE 21     /* for symbolic & numeric */
#define UMFPACK_FLOPS_ESTIMATE 22           /* flop count */
#define UMFPACK_LNZ_ESTIMATE 23             /* nz in L, incl. diagonal */
#define UMFPACK_UNZ_ESTIMATE 24             /* nz in U, incl. diagonal */
#define UMFPACK_VARIABLE_INIT_ESTIMATE 25   /* initial size of Numeric->Memory*/
#define UMFPACK_VARIABLE_PEAK_ESTIMATE 26   /* peak size of Numeric->Memory */
#define UMFPACK_VARIABLE_FINAL_ESTIMATE 27  /* final size of Numeric->Memory */
#define UMFPACK_MAX_FRONT_SIZE_ESTIMATE 28  /* max frontal matrix size */
#define UMFPACK_MAX_FRONT_NROWS_ESTIMATE 29 /* max # rows in any front */
#define UMFPACK_MAX_FRONT_NCOLS_ESTIMATE 30 /* max # columns in any front */

/* exact values, (estimates shown above) computed in UMFPACK_numeric: */
#define UMFPACK_NUMERIC_SIZE 40             /* final size of Numeric->Memory */
#define UMFPACK_PEAK_MEMORY 41              /* for symbolic & numeric */
#define UMFPACK_FLOPS 42                    /* flop count */
#define UMFPACK_LNZ 43                      /* nz in L, incl. diagonal */
#define UMFPACK_UNZ 44                      /* nz in U, incl. diagonal */
#define UMFPACK_VARIABLE_INIT 45            /* initial size of Numeric->Memory*/
#define UMFPACK_VARIABLE_PEAK 46            /* peak size of Numeric->Memory */
#define UMFPACK_VARIABLE_FINAL 47           /* final size of Numeric->Memory */
#define UMFPACK_MAX_FRONT_SIZE 48           /* max frontal matrix size */
#define UMFPACK_MAX_FRONT_NROWS 49          /* max # rows in any front */
#define UMFPACK_MAX_FRONT_NCOLS 50          /* max # columns in any front */

/* computed in UMFPACK_numeric: */
#define UMFPACK_NUMERIC_DEFRAG 60           /* # of garbage collections */
#define UMFPACK_NUMERIC_REALLOC 61          /* # of memory reallocations */
#define UMFPACK_NUMERIC_COSTLY_REALLOC 62   /* # of costlly memory realloc's */
#define UMFPACK_COMPRESSED_PATTERN 63       /* # of integers in LU pattern */
#define UMFPACK_LU_ENTRIES 64               /* # of reals in LU factors */
#define UMFPACK_NUMERIC_TIME 65             /* numeric factorization time */
#define UMFPACK_UDIAG_NZ 66                 /* nz on diagonal of U */
#define UMFPACK_RCOND 67                    /* est. reciprocal condition # */
#define UMFPACK_WAS_SCALED 68               /* none, max row, or sum row */
#define UMFPACK_RSMIN 69                    /* min (max row) or min (sum row) */
#define UMFPACK_RSMAX 70                    /* max (max row) or max (sum row) */
#define UMFPACK_UMIN 71                     /* min abs diagonal entry of U */
#define UMFPACK_UMAX 72                     /* max abs diagonal entry of U */
#define UMFPACK_ALLOC_INIT_USED 73          /* alloc_init parameter used */
#define UMFPACK_FORCED_UPDATES 74           /* # of forced updates */
#define UMFPACK_NUMERIC_WALLTIME 75         /* numeric wall clock time */
#define UMFPACK_NOFF_DIAG 76                /* number of off-diagonal pivots */

#define UMFPACK_ALL_LNZ 77                  /* nz in L, if no dropped entries */
#define UMFPACK_ALL_UNZ 78                  /* nz in U, if no dropped entries */
#define UMFPACK_NZDROPPED 79                /* # of dropped small entries */

/* computed in UMFPACK_solve: */
#define UMFPACK_IR_TAKEN 80         /* # of iterative refinement steps taken */
#define UMFPACK_IR_ATTEMPTED 81     /* # of iter. refinement steps attempted */
#define UMFPACK_OMEGA1 82           /* omega1, sparse backward error estimate */
#define UMFPACK_OMEGA2 83           /* omega2, sparse backward error estimate */
#define UMFPACK_SOLVE_FLOPS 84      /* flop count for solve */
#define UMFPACK_SOLVE_TIME 85       /* solve time (seconds) */
#define UMFPACK_SOLVE_WALLTIME 86   /* solve time (wall clock, seconds) */

/* Info [87, 88, 89] unused */

/* Unused parts of Info may be used in future versions of UMFPACK. */

/* -------------------------------------------------------------------------- */
/* contents of Control */
/* -------------------------------------------------------------------------- */

/* used in all UMFPACK_report_* routines: */
#define UMFPACK_PRL 0                   /* print level */

/* used in UMFPACK_*symbolic only: */
#define UMFPACK_DENSE_ROW 1             /* dense row parameter */
#define UMFPACK_DENSE_COL 2             /* dense col parameter */
#define UMFPACK_BLOCK_SIZE 4            /* BLAS-3 block size */
#define UMFPACK_STRATEGY 5              /* auto, symmetric, or unsym. */
#define UMFPACK_ORDERING 10             /* ordering method to use */
#define UMFPACK_FIXQ 13                 /* -1: no fixQ, 0: default, 1: fixQ */
#define UMFPACK_AMD_DENSE 14            /* for AMD ordering */
#define UMFPACK_AGGRESSIVE 19           /* whether or not to use aggressive */
#define UMFPACK_SINGLETONS 11           /* singleton filter on if true */

/* used in UMFPACK_numeric only: */
#define UMFPACK_PIVOT_TOLERANCE 3       /* threshold partial pivoting setting */
#define UMFPACK_ALLOC_INIT 6            /* initial allocation ratio */
#define UMFPACK_SYM_PIVOT_TOLERANCE 15  /* threshold, only for diag. entries */
#define UMFPACK_SCALE 16                /* what row scaling to do */
#define UMFPACK_FRONT_ALLOC_INIT 17     /* frontal matrix allocation ratio */
#define UMFPACK_DROPTOL 18              /* drop tolerance for entries in L,U */

/* used in UMFPACK_*solve only: */
#define UMFPACK_IRSTEP 7                /* max # of iterative refinements */

/* compile-time settings - Control [8..11] cannot be changed at run time: */
#define UMFPACK_COMPILED_WITH_BLAS 8        /* uses the BLAS */

// strategy control (added for v6.0.0)
#define UMFPACK_STRATEGY_THRESH_SYM 9          /* symmetry threshold */
#define UMFPACK_STRATEGY_THRESH_NNZDIAG 12     /* nnz(diag(A)) threshold */

/* -------------------------------------------------------------------------- */

/* Control [UMFPACK_STRATEGY] is one of the following: */
#define UMFPACK_STRATEGY_AUTO 0         /* use sym. or unsym. strategy */
#define UMFPACK_STRATEGY_UNSYMMETRIC 1  /* COLAMD(A), coletree postorder,
                                           not prefer diag*/
#define UMFPACK_STRATEGY_OBSOLETE 2     /* 2-by-2 is no longer available */
#define UMFPACK_STRATEGY_SYMMETRIC 3    /* AMD(A+A'), no coletree postorder,
                                           prefer diagonal */

/* Control [UMFPACK_SCALE] is one of the following: */
#define UMFPACK_SCALE_NONE 0    /* no scaling */
#define UMFPACK_SCALE_SUM 1     /* default: divide each row by sum (abs (row))*/
#define UMFPACK_SCALE_MAX 2     /* divide each row by max (abs (row)) */

/* Control [UMFPACK_ORDERING] and Info [UMFPACK_ORDERING_USED] are one of: */
#define UMFPACK_ORDERING_CHOLMOD 0      /* use CHOLMOD (AMD/COLAMD then METIS)*/
#define UMFPACK_ORDERING_AMD 1          /* use AMD/COLAMD */
#define UMFPACK_ORDERING_GIVEN 2        /* user-provided Qinit */
#define UMFPACK_ORDERING_METIS 3        /* use METIS */
#define UMFPACK_ORDERING_BEST 4         /* try many orderings, pick best */
#define UMFPACK_ORDERING_NONE 5         /* natural ordering */
#define UMFPACK_ORDERING_USER 6         /* user-provided function */

// ordering option added for v6.0.0:
#define UMFPACK_ORDERING_METIS_GUARD 7  // Use METIS, AMD, or COLAMD.
    // Symmetric strategy: always use METIS on A+A'.  Unsymmetric strategy: use
    // METIS on A'A, unless A has one or more very dense rows.  In that case,
    // A'A is very costly to form.  In this case, COLAMD is used instead of
    // METIS.

/* AMD/COLAMD means: use AMD for symmetric strategy, COLAMD for unsymmetric */

/* -------------------------------------------------------------------------- */
/* default values of Control: */
/* -------------------------------------------------------------------------- */

#define UMFPACK_DEFAULT_PRL 1
#define UMFPACK_DEFAULT_DENSE_ROW 0.2
#define UMFPACK_DEFAULT_DENSE_COL 0.2
#define UMFPACK_DEFAULT_PIVOT_TOLERANCE 0.1
#define UMFPACK_DEFAULT_SYM_PIVOT_TOLERANCE 0.001
#define UMFPACK_DEFAULT_BLOCK_SIZE 32
#define UMFPACK_DEFAULT_ALLOC_INIT 0.7
#define UMFPACK_DEFAULT_FRONT_ALLOC_INIT 0.5
#define UMFPACK_DEFAULT_IRSTEP 2
#define UMFPACK_DEFAULT_SCALE UMFPACK_SCALE_SUM
#define UMFPACK_DEFAULT_STRATEGY UMFPACK_STRATEGY_AUTO
#define UMFPACK_DEFAULT_AMD_DENSE AMD_DEFAULT_DENSE
#define UMFPACK_DEFAULT_FIXQ 0
#define UMFPACK_DEFAULT_AGGRESSIVE 1
#define UMFPACK_DEFAULT_DROPTOL 0
#define UMFPACK_DEFAULT_ORDERING UMFPACK_ORDERING_AMD
#define UMFPACK_DEFAULT_SINGLETONS TRUE

// added for v6.0.0.  Default changed fro 0.5 to 0.3
#define UMFPACK_DEFAULT_STRATEGY_THRESH_SYM 0.3         /* was 0.5 */
#define UMFPACK_DEFAULT_STRATEGY_THRESH_NNZDIAG 0.9

/* default values of Control may change in future versions of UMFPACK. */

/* -------------------------------------------------------------------------- */
/* status codes */
/* -------------------------------------------------------------------------- */

#define UMFPACK_OK (0)

/* status > 0 means a warning, but the method was successful anyway. */
/* A Symbolic or Numeric object was still created. */
#define UMFPACK_WARNING_singular_matrix (1)

/* The following warnings were added in umfpack_*_get_determinant */
#define UMFPACK_WARNING_determinant_underflow (2)
#define UMFPACK_WARNING_determinant_overflow (3)

/* status < 0 means an error, and the method was not successful. */
/* No Symbolic of Numeric object was created. */
#define UMFPACK_ERROR_out_of_memory (-1)
#define UMFPACK_ERROR_invalid_Numeric_object (-3)
#define UMFPACK_ERROR_invalid_Symbolic_object (-4)
#define UMFPACK_ERROR_argument_missing (-5)
#define UMFPACK_ERROR_n_nonpositive (-6)
#define UMFPACK_ERROR_invalid_matrix (-8)
#define UMFPACK_ERROR_different_pattern (-11)
#define UMFPACK_ERROR_invalid_system (-13)
#define UMFPACK_ERROR_invalid_permutation (-15)
#define UMFPACK_ERROR_internal_error (-911) /* yes, call me if you get this! */
#define UMFPACK_ERROR_file_IO (-17)

#define UMFPACK_ERROR_ordering_failed (-18)
#define UMFPACK_ERROR_invalid_blob (-19)

/* -------------------------------------------------------------------------- */
/* solve codes */
/* -------------------------------------------------------------------------- */

/* Solve the system ( )x=b, where ( ) is defined below.  "t" refers to the */
/* linear algebraic transpose (complex conjugate if A is complex), or the (') */
/* operator in MATLAB.  "at" refers to the array transpose, or the (.') */
/* operator in MATLAB. */

#define UMFPACK_A       (0)     /* Ax=b    */
#define UMFPACK_At      (1)     /* A'x=b   */
#define UMFPACK_Aat     (2)     /* A.'x=b  */

#define UMFPACK_Pt_L    (3)     /* P'Lx=b  */
#define UMFPACK_L       (4)     /* Lx=b    */
#define UMFPACK_Lt_P    (5)     /* L'Px=b  */
#define UMFPACK_Lat_P   (6)     /* L.'Px=b */
#define UMFPACK_Lt      (7)     /* L'x=b   */
#define UMFPACK_Lat     (8)     /* L.'x=b  */

#define UMFPACK_U_Qt    (9)     /* UQ'x=b  */
#define UMFPACK_U       (10)    /* Ux=b    */
#define UMFPACK_Q_Ut    (11)    /* QU'x=b  */
#define UMFPACK_Q_Uat   (12)    /* QU.'x=b */
#define UMFPACK_Ut      (13)    /* U'x=b   */
#define UMFPACK_Uat     (14)    /* U.'x=b  */

/* Integer constants are used for status and solve codes instead of enum */
/* to make it easier for a Fortran code to call UMFPACK. */


//==============================================================================
//==== USER-CALLABLE ROUTINES ==================================================
//==============================================================================


//==============================================================================
//==== Primary routines ========================================================
//==============================================================================

//------------------------------------------------------------------------------
// umfpack_symbolic
//------------------------------------------------------------------------------

int umfpack_di_symbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_dl_symbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_symbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zl_symbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int32_t n_row, n_col, *Ap, *Ai ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax ;
    int status = umfpack_di_symbolic (n_row, n_col, Ap, Ai, Ax,
        &Symbolic, Control, Info) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int64_t n_row, n_col, *Ap, *Ai ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax ;
    int status = umfpack_dl_symbolic (n_row, n_col, Ap, Ai, Ax,
        &Symbolic, Control, Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int32_t n_row, n_col, *Ap, *Ai ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax, *Az ;
    int status = umfpack_zi_symbolic (n_row, n_col, Ap, Ai, Ax, Az,
        &Symbolic, Control, Info) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int64_t n_row, n_col, *Ap, *Ai ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax, *Az ;
    int status = umfpack_zl_symbolic (n_row, n_col, Ap, Ai, Ax, Az,
        &Symbolic, Control, Info) ;

packed complex Syntax:

    Same as above, except Az is NULL.

Purpose:

    Given nonzero pattern of a sparse matrix A in column-oriented form,
    umfpack_*_symbolic performs a column pre-ordering to reduce fill-in
    (using COLAMD, AMD or METIS) and a symbolic factorization.  This is required
    before the matrix can be numerically factorized with umfpack_*_numeric.
    If you wish to bypass the COLAMD/AMD/METIS pre-ordering and provide your own
    ordering, use umfpack_*_qsymbolic instead.  If you wish to pass in a
    pointer to a user-provided ordering function, use umfpack_*_fsymbolic.

    Since umfpack_*_symbolic and umfpack_*_qsymbolic are very similar, options
    for both routines are discussed below.

    For the following discussion, let S be the submatrix of A obtained after
    eliminating all pivots of zero Markowitz cost.  S has dimension
    (n_row-n1-nempty_row) -by- (n_col-n1-nempty_col), where
    n1 = Info [UMFPACK_COL_SINGLETONS] + Info [UMFPACK_ROW_SINGLETONS],
    nempty_row = Info [UMFPACK_NEMPTY_ROW] and
    nempty_col = Info [UMFPACK_NEMPTY_COL].

Returns:

    The status code is returned.  See Info [UMFPACK_STATUS], below.

Arguments:

    Int n_row ;         Input argument, not modified.
    Int n_col ;         Input argument, not modified.

        A is an n_row-by-n_col matrix.  Restriction: n_row > 0 and n_col > 0.

    Int Ap [n_col+1] ;  Input argument, not modified.

        Ap is an integer array of size n_col+1.  On input, it holds the
        "pointers" for the column form of the sparse matrix A.  Column j of
        the matrix A is held in Ai [(Ap [j]) ... (Ap [j+1]-1)].  The first
        entry, Ap [0], must be zero, and Ap [j] <= Ap [j+1] must hold for all
        j in the range 0 to n_col-1.  The value nz = Ap [n_col] is thus the
        total number of entries in the pattern of the matrix A.  nz must be
        greater than or equal to zero.

    Int Ai [nz] ;       Input argument, not modified, of size nz = Ap [n_col].

        The nonzero pattern (row indices) for column j is stored in
        Ai [(Ap [j]) ... (Ap [j+1]-1)].  The row indices in a given column j
        must be in ascending order, and no duplicate row indices may be present.
        Row indices must be in the range 0 to n_row-1 (the matrix is 0-based).
        See umfpack_*_triplet_to_col for how to sort the columns of a matrix
        and sum up the duplicate entries.  See umfpack_*_report_matrix for how
        to print the matrix A.

    double Ax [nz] ;    Optional input argument, not modified.  May be NULL.
                        Size 2*nz for packed complex case.

        The numerical values of the sparse matrix A.  The nonzero pattern (row
        indices) for column j is stored in Ai [(Ap [j]) ... (Ap [j+1]-1)], and
        the corresponding numerical values are stored in
        Ax [(Ap [j]) ... (Ap [j+1]-1)].  Used only for gathering statistics
        about how many nonzeros are placed on the diagonal by the fill-reducing
        ordering.

    double Az [nz] ;    Optional input argument, not modified, for complex
                        versions.  May be NULL.

        For the complex versions, this holds the imaginary part of A.  The
        imaginary part of column j is held in Az [(Ap [j]) ... (Ap [j+1]-1)].

        If Az is NULL, then both real
        and imaginary parts are contained in Ax[0..2*nz-1], with Ax[2*k]
        and Ax[2*k+1] being the real and imaginary part of the kth entry.

        Used for statistics only.  See the description of Ax, above.

    void **Symbolic ;   Output argument.

        **Symbolic is the address of a (void *) pointer variable in the user's
        calling routine (see Syntax, above).  On input, the contents of this
        variable are not defined.  On output, this variable holds a (void *)
        pointer to the Symbolic object (if successful), or (void *) NULL if
        a failure occurred.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used (the defaults are suitable for all matrices,
        ranging from those with highly unsymmetric nonzero pattern, to
        symmetric matrices).  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_STRATEGY]:  This is the most important control
            parameter.  It determines what kind of ordering and pivoting
            strategy that UMFPACK should use.

            NOTE: the interaction of numerical and fill-reducing pivoting is
            a delicate balance, and a perfect hueristic is not possible because
            sparsity-preserving pivoting is an NP-hard problem.  Selecting the
            wrong strategy can lead to catastrophic fill-in and/or numerical
            inaccuracy.

            UMFPACK_STRATEGY_AUTO:  This is the default.  The input matrix is
                analyzed to determine how symmetric the nonzero pattern is, and
                how many entries there are on the diagonal.  It then selects one
                of the following strategies.  Refer to the User Guide for a
                description of how the strategy is automatically selected.

            UMFPACK_STRATEGY_UNSYMMETRIC:  Use the unsymmetric strategy.  COLAMD
                is used to order the columns of A, followed by a postorder of
                the column elimination tree.  No attempt is made to perform
                diagonal pivoting.  The column ordering is refined during
                factorization.

                In the numerical factorization, the
                Control [UMFPACK_SYM_PIVOT_TOLERANCE] parameter is ignored.  A
                pivot is selected if its magnitude is >=
                Control [UMFPACK_PIVOT_TOLERANCE] (default 0.1) times the
                largest entry in its column.

            UMFPACK_STRATEGY_SYMMETRIC:  Use the symmetric strategy
                In this method, the approximate minimum degree
                ordering (AMD) is applied to A+A', followed by a postorder of
                the elimination tree of A+A'.  UMFPACK attempts to perform
                diagonal pivoting during numerical factorization.  No refinement
                of the column pre-ordering is performed during factorization.

                In the numerical factorization, a nonzero entry on the diagonal
                is selected as the pivot if its magnitude is >= Control
                [UMFPACK_SYM_PIVOT_TOLERANCE] (default 0.001) times the largest
                entry in its column.  If this is not acceptable, then an
                off-diagonal pivot is selected with magnitude >= Control
                [UMFPACK_PIVOT_TOLERANCE] (default 0.1) times the largest entry
                in its column.

        Control [UMFPACK_ORDERING]:  The ordering method to use:
            UMFPACK_ORDERING_CHOLMOD    try AMD/COLAMD, then METIS if needed
            UMFPACK_ORDERING_AMD        just AMD or COLAMD
            UMFPACK_ORDERING_GIVEN      just Qinit (umfpack_*_qsymbolic only)
            UMFPACK_ORDERING_NONE       no fill-reducing ordering
            UMFPACK_ORDERING_METIS      just METIS(A+A') or METIS(A'A)
            UMFPACK_ORDERING_BEST       try AMD/COLAMD, METIS, and NESDIS
            UMFPACK_ORDERING_USER       just user function (*_fsymbolic only)
            UMFPACK_ORDERING_METIS_GUARD use METIS, AMD, or COLAMD.
                Symmetric strategy: always use METIS on A+A'.  Unsymmetric
                strategy: use METIS on A'A, unless A has one or more very dense
                rows.  In that case, A'A is very costly to form.  In this case,
                COLAMD is used instead of METIS.

        Control [UMFPACK_SINGLETONS]: If false (0), then singletons are
            not removed prior to factorization.  Default: true (1).

        Control [UMFPACK_DENSE_COL]:
            If COLAMD is used, columns with more than
            max (16, Control [UMFPACK_DENSE_COL] * 16 * sqrt (n_row)) entries
            are placed placed last in the column pre-ordering.  Default: 0.2.

        Control [UMFPACK_DENSE_ROW]:
            Rows with more than max (16, Control [UMFPACK_DENSE_ROW] * 16 *
            sqrt (n_col)) entries are treated differently in the COLAMD
            pre-ordering, and in the internal data structures during the
            subsequent numeric factorization.  Default: 0.2.
            If any row exists with more than these number of entries, and
            if the unsymmetric strategy is selected, the METIS_GUARD ordering
            selects COLAMD instead of METIS.

        Control [UMFPACK_AMD_DENSE]:  rows/columns in A+A' with more than
            max (16, Control [UMFPACK_AMD_DENSE] * sqrt (n)) entries
            (where n = n_row = n_col) are ignored in the AMD pre-ordering.
            Default: 10.

        Control [UMFPACK_BLOCK_SIZE]:  the block size to use for Level-3 BLAS
            in the subsequent numerical factorization (umfpack_*_numeric).
            A value less than 1 is treated as 1.  Default: 32.  Modifying this
            parameter affects when updates are applied to the working frontal
            matrix, and can indirectly affect fill-in and operation count.
            Assuming the block size is large enough (8 or so), this parameter
            has a modest effect on performance.

        Control [UMFPACK_FIXQ]:  If > 0, then the pre-ordering Q is not modified
            during numeric factorization.  If < 0, then Q may be modified.  If
            zero, then this is controlled automatically (the unsymmetric
            strategy modifies Q, the others do not).  Default: 0.

            Note that the symbolic analysis will in general modify the input
            ordering Qinit to obtain Q; see umfpack_qsymbolic.h for details.
            This option ensures Q does not change, as found in the symbolic
            analysis, but Qinit is in general not the same as Q.

        Control [UMFPACK_AGGRESSIVE]:  If nonzero, aggressive absorption is used
            in COLAMD and AMD.  Default: 1.

        // added for v6.0.0:
        Control [UMFPACK_STRATEGY_THRESH_SYM]: tsym, Default 0.5.
        Control [UMFPACK_STRATEGY_THRESH_NNZDIAG]: tdiag, Default 0.9.
            For the auto strategy, if the pattern of the submatrix S after
            removing singletons has a symmetry of tsym or more (0 being
            completely unsymmetric and 1 being completely symmetric, and if the
            fraction of entries present on the diagonal is >= tdiag, then the
            symmetric strategy is chosen.  Otherwise, the unsymmetric strategy
            is chosen.

    double Info [UMFPACK_INFO] ;        Output argument, not defined on input.

        Contains statistics about the symbolic analysis.  If a (double *) NULL
        pointer is passed, then no statistics are returned in Info (this is not
        an error condition).  The entire Info array is cleared (all entries set
        to -1) and then the following statistics are computed:

        Info [UMFPACK_STATUS]: status code.  This is also the return value,
            whether or not Info is present.

            UMFPACK_OK

                Each column of the input matrix contained row indices
                in increasing order, with no duplicates.  Only in this case
                does umfpack_*_symbolic compute a valid symbolic factorization.
                For the other cases below, no Symbolic object is created
                (*Symbolic is (void *) NULL).

            UMFPACK_ERROR_n_nonpositive

                n is less than or equal to zero.

            UMFPACK_ERROR_invalid_matrix

                Number of entries in the matrix is negative, Ap [0] is nonzero,
                a column has a negative number of entries, a row index is out of
                bounds, or the columns of input matrix were jumbled (unsorted
                columns or duplicate entries).

            UMFPACK_ERROR_out_of_memory

                Insufficient memory to perform the symbolic analysis.  If the
                analysis requires more than 2GB of memory and you are using
                the int32_t version of UMFPACK, then you are guaranteed
                to run out of memory.  Try using the 64-bit version of UMFPACK.

            UMFPACK_ERROR_argument_missing

                One or more required arguments is missing.

            UMFPACK_ERROR_internal_error

                Something very serious went wrong.  This is a bug.
                Please contact the author (DrTimothyAldenDavis@gmail.com).

        Info [UMFPACK_NROW]:  the value of the input argument n_row.

        Info [UMFPACK_NCOL]:  the value of the input argument n_col.

        Info [UMFPACK_NZ]:  the number of entries in the input matrix
            (Ap [n_col]).

        Info [UMFPACK_SIZE_OF_UNIT]:  the number of bytes in a Unit,
            for memory usage statistics below.

        Info [UMFPACK_SIZE_OF_INT]:  the number of bytes in an int32_t.

        Info [UMFPACK_SIZE_OF_LONG]:  the number of bytes in a int64_t.

        Info [UMFPACK_SIZE_OF_POINTER]:  the number of bytes in a void *
            pointer.

        Info [UMFPACK_SIZE_OF_ENTRY]:  the number of bytes in a numerical entry.

        Info [UMFPACK_NDENSE_ROW]:  number of "dense" rows in A.  These rows are
            ignored when the column pre-ordering is computed in COLAMD.  They
            are also treated differently during numeric factorization.  If > 0,
            then the matrix had to be re-analyzed by UMF_analyze, which does
            not ignore these rows.

        Info [UMFPACK_NEMPTY_ROW]:  number of "empty" rows in A, as determined
            These are rows that either have no entries, or whose entries are
            all in pivot columns of zero-Markowitz-cost pivots.

        Info [UMFPACK_NDENSE_COL]:  number of "dense" columns in A.  COLAMD
            orders these columns are ordered last in the factorization, but
            before "empty" columns.

        Info [UMFPACK_NEMPTY_COL]:  number of "empty" columns in A.  These are
            columns that either have no entries, or whose entries are all in
            pivot rows of zero-Markowitz-cost pivots.  These columns are
            ordered last in the factorization, to the right of "dense" columns.

        Info [UMFPACK_SYMBOLIC_DEFRAG]:  number of garbage collections
            performed during ordering and symbolic pre-analysis.

        Info [UMFPACK_SYMBOLIC_PEAK_MEMORY]:  the amount of memory (in Units)
            required for umfpack_*_symbolic to complete.  This count includes
            the size of the Symbolic object itself, which is also reported in
            Info [UMFPACK_SYMBOLIC_SIZE].

        Info [UMFPACK_SYMBOLIC_SIZE]: the final size of the Symbolic object (in
            Units).  This is fairly small, roughly 2*n to 13*n integers,
            depending on the matrix.

        Info [UMFPACK_VARIABLE_INIT_ESTIMATE]: the Numeric object contains two
            parts.  The first is fixed in size (O (n_row+n_col)).  The
            second part holds the sparse LU factors and the contribution blocks
            from factorized frontal matrices.  This part changes in size during
            factorization.  Info [UMFPACK_VARIABLE_INIT_ESTIMATE] is the exact
            size (in Units) required for this second variable-sized part in
            order for the numerical factorization to start.

        Info [UMFPACK_VARIABLE_PEAK_ESTIMATE]: the estimated peak size (in
            Units) of the variable-sized part of the Numeric object.  This is
            usually an upper bound, but that is not guaranteed.

        Info [UMFPACK_VARIABLE_FINAL_ESTIMATE]: the estimated final size (in
            Units) of the variable-sized part of the Numeric object.  This is
            usually an upper bound, but that is not guaranteed.  It holds just
            the sparse LU factors.

        Info [UMFPACK_NUMERIC_SIZE_ESTIMATE]:  an estimate of the final size (in
            Units) of the entire Numeric object (both fixed-size and variable-
            sized parts), which holds the LU factorization (including the L, U,
            P and Q matrices).

        Info [UMFPACK_PEAK_MEMORY_ESTIMATE]:  an estimate of the total amount of
            memory (in Units) required by umfpack_*_symbolic and
            umfpack_*_numeric to perform both the symbolic and numeric
            factorization.  This is the larger of the amount of memory needed
            in umfpack_*_numeric itself, and the amount of memory needed in
            umfpack_*_symbolic (Info [UMFPACK_SYMBOLIC_PEAK_MEMORY]).  The
            count includes the size of both the Symbolic and Numeric objects
            themselves.  It can be a very loose upper bound, particularly when
            the symmetric strategy is used.

        Info [UMFPACK_FLOPS_ESTIMATE]:  an estimate of the total floating-point
            operations required to factorize the matrix.  This is a "true"
            theoretical estimate of the number of flops that would be performed
            by a flop-parsimonious sparse LU algorithm.  It assumes that no
            extra flops are performed except for what is strictly required to
            compute the LU factorization.  It ignores, for example, the flops
            performed by umfpack_di_numeric to add contribution blocks of
            frontal matrices together.  If L and U are the upper bound on the
            pattern of the factors, then this flop count estimate can be
            represented in MATLAB (for real matrices, not complex) as:

                Lnz = full (sum (spones (L))) - 1 ;     % nz in each col of L
                Unz = full (sum (spones (U')))' - 1 ;   % nz in each row of U
                flops = 2*Lnz*Unz + sum (Lnz) ;

            The actual "true flop" count found by umfpack_*_numeric will be
            less than this estimate.

            For the real version, only (+ - * /) are counted.  For the complex
            version, the following counts are used:

                operation       flops
                c = 1/b         6
                c = a*b         6
                c -= a*b        8

        Info [UMFPACK_LNZ_ESTIMATE]:  an estimate of the number of nonzeros in
            L, including the diagonal.  Since L is unit-diagonal, the diagonal
            of L is not stored.  This estimate is a strict upper bound on the
            actual nonzeros in L to be computed by umfpack_*_numeric.

        Info [UMFPACK_UNZ_ESTIMATE]:  an estimate of the number of nonzeros in
            U, including the diagonal.  This estimate is a strict upper bound on
            the actual nonzeros in U to be computed by umfpack_*_numeric.

        Info [UMFPACK_MAX_FRONT_SIZE_ESTIMATE]: estimate of the size of the
            largest frontal matrix (# of entries), for arbitrary partial
            pivoting during numerical factorization.

        Info [UMFPACK_SYMBOLIC_TIME]:  The CPU time taken, in seconds.

        Info [UMFPACK_SYMBOLIC_WALLTIME]:  The wallclock time taken, in seconds.

        Info [UMFPACK_STRATEGY_USED]: The ordering strategy used:
            UMFPACK_STRATEGY_SYMMETRIC or UMFPACK_STRATEGY_UNSYMMETRIC

        Info [UMFPACK_ORDERING_USED]:  The ordering method used:
            UMFPACK_ORDERING_AMD    (AMD for sym. strategy, COLAMD for unsym.)
            UMFPACK_ORDERING_GIVEN
            UMFPACK_ORDERING_NONE
            UMFPACK_ORDERING_METIS
            UMFPACK_ORDERING_USER

        Info [UMFPACK_QFIXED]: 1 if the column pre-ordering will be refined
            during numerical factorization, 0 if not.

        Info [UMFPACK_DIAG_PREFERED]: 1 if diagonal pivoting will be attempted,
            0 if not.

        Info [UMFPACK_COL_SINGLETONS]:  the matrix A is analyzed by first
            eliminating all pivots with zero Markowitz cost.  This count is the
            number of these pivots with exactly one nonzero in their pivot
            column.

        Info [UMFPACK_ROW_SINGLETONS]:  the number of zero-Markowitz-cost
            pivots with exactly one nonzero in their pivot row.

        Info [UMFPACK_PATTERN_SYMMETRY]: the symmetry of the pattern of S.

        Info [UMFPACK_NZ_A_PLUS_AT]: the number of off-diagonal entries in S+S'.

        Info [UMFPACK_NZDIAG]:  the number of entries on the diagonal of S.

        Info [UMFPACK_N2]:  if S is square, and nempty_row = nempty_col, this
            is equal to n_row - n1 - nempty_row.

        Info [UMFPACK_S_SYMMETRIC]: 1 if S is square and its diagonal has been
            preserved, 0 otherwise.


        Info [UMFPACK_MAX_FRONT_NROWS_ESTIMATE]: estimate of the max number of
            rows in any frontal matrix, for arbitrary partial pivoting.

        Info [UMFPACK_MAX_FRONT_NCOLS_ESTIMATE]: estimate of the max number of
            columns in any frontal matrix, for arbitrary partial pivoting.

        ------------------------------------------------------------------------
        The next four statistics are computed only if AMD is used:
        ------------------------------------------------------------------------

        Info [UMFPACK_SYMMETRIC_LUNZ]: The number of nonzeros in L and U,
            assuming no pivoting during numerical factorization, and assuming a
            zero-free diagonal of U.  Excludes the entries on the diagonal of
            L.  If the matrix has a purely symmetric nonzero pattern, this is
            often a lower bound on the nonzeros in the actual L and U computed
            in the numerical factorization, for matrices that fit the criteria
            for the "symmetric" strategy.

        Info [UMFPACK_SYMMETRIC_FLOPS]: The floating-point operation count in
            the numerical factorization phase, assuming no pivoting.  If the
            pattern of the matrix is symmetric, this is normally a lower bound
            on the floating-point operation count in the actual numerical
            factorization, for matrices that fit the criteria for the symmetric
            strategy.

        Info [UMFPACK_SYMMETRIC_NDENSE]: The number of "dense" rows/columns of
            S+S' that were ignored during the AMD ordering.  These are placed
            last in the output order.  If > 0, then the
            Info [UMFPACK_SYMMETRIC_*] statistics, above are rough upper bounds.

        Info [UMFPACK_SYMMETRIC_DMAX]: The maximum number of nonzeros in any
            column of L, if no pivoting is performed during numerical
            factorization.  Excludes the part of the LU factorization for
            pivots with zero Markowitz cost.

        At the start of umfpack_*_symbolic, all of Info is set of -1, and then
        after that only the above listed Info [...] entries are accessed.
        Future versions might modify different parts of Info.
*/

//------------------------------------------------------------------------------
// umfpack_numeric
//------------------------------------------------------------------------------

int umfpack_di_numeric
(
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    void *Symbolic,
    void **Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_dl_numeric
(
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    void *Symbolic,
    void **Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_numeric
(
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    void *Symbolic,
    void **Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zl_numeric
(
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    void *Symbolic,
    void **Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic, *Numeric ;
    int32_t *Ap, *Ai, status ;
    double *Ax, Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    int status = umfpack_di_numeric (Ap, Ai, Ax, Symbolic, &Numeric, Control,
        Info) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic, *Numeric ;
    int64_t *Ap, *Ai ;
    double *Ax, Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    int status = umfpack_dl_numeric (Ap, Ai, Ax, Symbolic, &Numeric, Control,
        Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic, *Numeric ;
    int32_t *Ap, *Ai ;
    double *Ax, *Az, Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    int status = umfpack_zi_numeric (Ap, Ai, Ax, Az, Symbolic, &Numeric,
        Control, Info) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic, *Numeric ;
    int64_t *Ap, *Ai ;
    double *Ax, *Az, Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    int status = umfpack_zl_numeric (Ap, Ai, Ax, Az, Symbolic, &Numeric,
        Control, Info) ;

packed complex Syntax:

    Same as above, except that Az is NULL.

Purpose:

    Given a sparse matrix A in column-oriented form, and a symbolic analysis
    computed by umfpack_*_*symbolic, the umfpack_*_numeric routine performs the
    numerical factorization, PAQ=LU, PRAQ=LU, or P(R\A)Q=LU, where P and Q are
    permutation matrices (represented as permutation vectors), R is the row
    scaling, L is unit-lower triangular, and U is upper triangular.  This is
    required before the system Ax=b (or other related linear systems) can be
    solved.  umfpack_*_numeric can be called multiple times for each call to
    umfpack_*_*symbolic, to factorize a sequence of matrices with identical
    nonzero pattern.  Simply compute the Symbolic object once, with
    umfpack_*_*symbolic, and reuse it for subsequent matrices.  This routine
    safely detects if the pattern changes, and sets an appropriate error code.

Returns:

    The status code is returned.  See Info [UMFPACK_STATUS], below.

Arguments:

    Int Ap [n_col+1] ;  Input argument, not modified.

        This must be identical to the Ap array passed to umfpack_*_*symbolic.
        The value of n_col is what was passed to umfpack_*_*symbolic (this is
        held in the Symbolic object).

    Int Ai [nz] ;       Input argument, not modified, of size nz = Ap [n_col].

        This must be identical to the Ai array passed to umfpack_*_*symbolic.

    double Ax [nz] ;    Input argument, not modified, of size nz = Ap [n_col].
                        Size 2*nz for packed complex case.

        The numerical values of the sparse matrix A.  The nonzero pattern (row
        indices) for column j is stored in Ai [(Ap [j]) ... (Ap [j+1]-1)], and
        the corresponding numerical values are stored in
        Ax [(Ap [j]) ... (Ap [j+1]-1)].

    double Az [nz] ;    Input argument, not modified, for complex versions.

        For the complex versions, this holds the imaginary part of A.  The
        imaginary part of column j is held in Az [(Ap [j]) ... (Ap [j+1]-1)].

        If Az is NULL, then both real
        and imaginary parts are contained in Ax[0..2*nz-1], with Ax[2*k]
        and Ax[2*k+1] being the real and imaginary part of the kth entry.

    void *Symbolic ;    Input argument, not modified.

        The Symbolic object, which holds the symbolic factorization computed by
        umfpack_*_*symbolic.  The Symbolic object is not modified by
        umfpack_*_numeric.

    void **Numeric ;    Output argument.

        **Numeric is the address of a (void *) pointer variable in the user's
        calling routine (see Syntax, above).  On input, the contents of this
        variable are not defined.  On output, this variable holds a (void *)
        pointer to the Numeric object (if successful), or (void *) NULL if
        a failure occurred.

    double Control [UMFPACK_CONTROL] ;   Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PIVOT_TOLERANCE]:  relative pivot tolerance for
            threshold partial pivoting with row interchanges.  In any given
            column, an entry is numerically acceptable if its absolute value is
            greater than or equal to Control [UMFPACK_PIVOT_TOLERANCE] times
            the largest absolute value in the column.  A value of 1.0 gives true
            partial pivoting.  If less than or equal to zero, then any nonzero
            entry is numerically acceptable as a pivot.  Default: 0.1.

            Smaller values tend to lead to sparser LU factors, but the solution
            to the linear system can become inaccurate.  Larger values can lead
            to a more accurate solution (but not always), and usually an
            increase in the total work.

            For complex matrices, a cheap approximate of the absolute value
            is used for the threshold partial pivoting test (|a_real| + |a_imag|
            instead of the more expensive-to-compute exact absolute value
            sqrt (a_real^2 + a_imag^2)).

        Control [UMFPACK_SYM_PIVOT_TOLERANCE]:
            If diagonal pivoting is attempted (the symmetric
            strategy is used) then this parameter is used to control when the
            diagonal entry is selected in a given pivot column.  The absolute
            value of the entry must be >= Control [UMFPACK_SYM_PIVOT_TOLERANCE]
            times the largest absolute value in the column.  A value of zero
            will ensure that no off-diagonal pivoting is performed, except that
            zero diagonal entries are not selected if there are any off-diagonal
            nonzero entries.

            If an off-diagonal pivot is selected, an attempt is made to restore
            symmetry later on.  Suppose A (i,j) is selected, where i != j.
            If column i has not yet been selected as a pivot column, then
            the entry A (j,i) is redefined as a "diagonal" entry, except that
            the tighter tolerance (Control [UMFPACK_PIVOT_TOLERANCE]) is
            applied.  This strategy has an effect similar to 2-by-2 pivoting
            for symmetric indefinite matrices.  If a 2-by-2 block pivot with
            nonzero structure

                       i j
                    i: 0 x
                    j: x 0

            is selected in a symmetric indefinite factorization method, the
            2-by-2 block is inverted and a rank-2 update is applied.  In
            UMFPACK, this 2-by-2 block would be reordered as

                       j i
                    i: x 0
                    j: 0 x

            In both cases, the symmetry of the Schur complement is preserved.

        Control [UMFPACK_SCALE]:  Note that the user's input matrix is
            never modified, only an internal copy is scaled.

            There are three valid settings for this parameter.  If any other
            value is provided, the default is used.

            UMFPACK_SCALE_NONE:  no scaling is performed.

            UMFPACK_SCALE_SUM:  each row of the input matrix A is divided by
                the sum of the absolute values of the entries in that row.
                The scaled matrix has an infinity norm of 1.

            UMFPACK_SCALE_MAX:  each row of the input matrix A is divided by
                the maximum the absolute values of the entries in that row.
                In the scaled matrix the largest entry in each row has
                a magnitude exactly equal to 1.

            Note that for complex matrices, a cheap approximate absolute value
            is used, |a_real| + |a_imag|, instead of the exact absolute value
            sqrt ((a_real)^2 + (a_imag)^2).

            Scaling is very important for the "symmetric" strategy when
            diagonal pivoting is attempted.  It also improves the performance
            of the "unsymmetric" strategy.

            Default: UMFPACK_SCALE_SUM.

        Control [UMFPACK_ALLOC_INIT]:

            When umfpack_*_numeric starts, it allocates memory for the Numeric
            object.  Part of this is of fixed size (approximately n double's +
            12*n integers).  The remainder is of variable size, which grows to
            hold the LU factors and the frontal matrices created during
            factorization.  A estimate of the upper bound is computed by
            umfpack_*_*symbolic, and returned by umfpack_*_*symbolic in
            Info [UMFPACK_VARIABLE_PEAK_ESTIMATE] (in Units).

            If Control [UMFPACK_ALLOC_INIT] is >= 0, umfpack_*_numeric initially
            allocates space for the variable-sized part equal to this estimate
            times Control [UMFPACK_ALLOC_INIT].  Typically, for matrices for
            which the "unsymmetric" strategy applies, umfpack_*_numeric needs
            only about half the estimated memory space, so a setting of 0.5 or
            0.6 often provides enough memory for umfpack_*_numeric to factorize
            the matrix with no subsequent increases in the size of this block.

            If the matrix is ordered via AMD, then this non-negative parameter
            is ignored.  The initial allocation ratio computed automatically,
            as 1.2 * (nz + Info [UMFPACK_SYMMETRIC_LUNZ]) /
            (Info [UMFPACK_LNZ_ESTIMATE] + Info [UMFPACK_UNZ_ESTIMATE] -
            min (n_row, n_col)).

            If Control [UMFPACK_ALLOC_INIT] is negative, then umfpack_*_numeric
            allocates a space with initial size (in Units) equal to
            (-Control [UMFPACK_ALLOC_INIT]).

            Regardless of the value of this parameter, a space equal to or
            greater than the the bare minimum amount of memory needed to start
            the factorization is always initially allocated.  The bare initial
            memory required is returned by umfpack_*_*symbolic in
            Info [UMFPACK_VARIABLE_INIT_ESTIMATE] (an exact value, not an
            estimate).

            If the variable-size part of the Numeric object is found to be too
            small sometime after numerical factorization has started, the memory
            is increased in size by a factor of 1.2.   If this fails, the
            request is reduced by a factor of 0.95 until it succeeds, or until
            it determines that no increase in size is possible.  Garbage
            collection then occurs.

            The strategy of attempting to "malloc" a working space, and
            re-trying with a smaller space, may not work when UMFPACK is used
            as a mexFunction MATLAB, since mxMalloc aborts the mexFunction if it
            fails.  This issue does not affect the use of UMFPACK as a part of
            the built-in x=A\b in MATLAB 6.5 and later.

            If you are using the umfpack mexFunction, decrease the magnitude of
            Control [UMFPACK_ALLOC_INIT] if you run out of memory in MATLAB.

            Default initial allocation size: 0.7.  Thus, with the default
            control settings and the "unsymmetric" strategy, the upper-bound is
            reached after two reallocations (0.7 * 1.2 * 1.2 = 1.008).

            Changing this parameter has little effect on fill-in or operation
            count.  It has a small impact on run-time (the extra time required
            to do the garbage collection and memory reallocation).

        Control [UMFPACK_FRONT_ALLOC_INIT]:

            When UMFPACK starts the factorization of each "chain" of frontal
            matrices, it allocates a working array to hold the frontal matrices
            as they are factorized.  The symbolic factorization computes the
            size of the largest possible frontal matrix that could occur during
            the factorization of each chain.

            If Control [UMFPACK_FRONT_ALLOC_INIT] is >= 0, the following
            strategy is used.  If the AMD ordering was used, this non-negative
            parameter is ignored.  A front of size (d+2)*(d+2) is allocated,
            where d = Info [UMFPACK_SYMMETRIC_DMAX].  Otherwise, a front of
            size Control [UMFPACK_FRONT_ALLOC_INIT] times the largest front
            possible for this chain is allocated.

            If Control [UMFPACK_FRONT_ALLOC_INIT] is negative, then a front of
            size (-Control [UMFPACK_FRONT_ALLOC_INIT]) is allocated (where the
            size is in terms of the number of numerical entries).  This is done
            regardless of the ordering method or ordering strategy used.

            Default: 0.5.

        Control [UMFPACK_DROPTOL]:

            Entries in L and U with absolute value less than or equal to the
            drop tolerance are removed from the data structures (unless leaving
            them there reduces memory usage by reducing the space required
            for the nonzero pattern of L and U).

            Default: 0.0.

    double Info [UMFPACK_INFO] ;        Output argument.

        Contains statistics about the numeric factorization.  If a
        (double *) NULL pointer is passed, then no statistics are returned in
        Info (this is not an error condition).  The following statistics are
        computed in umfpack_*_numeric:

        Info [UMFPACK_STATUS]: status code.  This is also the return value,
            whether or not Info is present.

            UMFPACK_OK

                Numeric factorization was successful.  umfpack_*_numeric
                computed a valid numeric factorization.

            UMFPACK_WARNING_singular_matrix

                Numeric factorization was successful, but the matrix is
                singular.  umfpack_*_numeric computed a valid numeric
                factorization, but you will get a divide by zero in
                umfpack_*_*solve.  For the other cases below, no Numeric object
                is created (*Numeric is (void *) NULL).

            UMFPACK_ERROR_out_of_memory

                Insufficient memory to complete the numeric factorization.

            UMFPACK_ERROR_argument_missing

                One or more required arguments are missing.

            UMFPACK_ERROR_invalid_Symbolic_object

                Symbolic object provided as input is invalid.

            UMFPACK_ERROR_different_pattern

                The pattern (Ap and/or Ai) has changed since the call to
                umfpack_*_*symbolic which produced the Symbolic object.

        Info [UMFPACK_NROW]:  the value of n_row stored in the Symbolic object.

        Info [UMFPACK_NCOL]:  the value of n_col stored in the Symbolic object.

        Info [UMFPACK_NZ]:  the number of entries in the input matrix.
            This value is obtained from the Symbolic object.

        Info [UMFPACK_SIZE_OF_UNIT]:  the number of bytes in a Unit, for memory
            usage statistics below.

        Info [UMFPACK_VARIABLE_INIT]: the initial size (in Units) of the
            variable-sized part of the Numeric object.  If this differs from
            Info [UMFPACK_VARIABLE_INIT_ESTIMATE], then the pattern (Ap and/or
            Ai) has changed since the last call to umfpack_*_*symbolic, which is
            an error condition.

        Info [UMFPACK_VARIABLE_PEAK]: the peak size (in Units) of the
            variable-sized part of the Numeric object.  This size is the amount
            of space actually used inside the block of memory, not the space
            allocated via UMF_malloc.  You can reduce UMFPACK's memory
            requirements by setting Control [UMFPACK_ALLOC_INIT] to the ratio
            Info [UMFPACK_VARIABLE_PEAK] / Info[UMFPACK_VARIABLE_PEAK_ESTIMATE].
            This will ensure that no memory reallocations occur (you may want to
            add 0.001 to make sure that integer roundoff does not lead to a
            memory size that is 1 Unit too small; otherwise, garbage collection
            and reallocation will occur).

        Info [UMFPACK_VARIABLE_FINAL]: the final size (in Units) of the
            variable-sized part of the Numeric object.  It holds just the
            sparse LU factors.

        Info [UMFPACK_NUMERIC_SIZE]:  the actual final size (in Units) of the
            entire Numeric object, including the final size of the variable
            part of the object.  Info [UMFPACK_NUMERIC_SIZE_ESTIMATE],
            an estimate, was computed by umfpack_*_*symbolic.  The estimate is
            normally an upper bound on the actual final size, but this is not
            guaranteed.

        Info [UMFPACK_PEAK_MEMORY]:  the actual peak memory usage (in Units) of
            both umfpack_*_*symbolic and umfpack_*_numeric.  An estimate,
            Info [UMFPACK_PEAK_MEMORY_ESTIMATE], was computed by
            umfpack_*_*symbolic.  The estimate is normally an upper bound on the
            actual peak usage, but this is not guaranteed.  With testing on
            hundreds of matrix arising in real applications, I have never
            observed a matrix where this estimate or the Numeric size estimate
            was less than the actual result, but this is theoretically possible.
            Please send me one if you find such a matrix.

        Info [UMFPACK_FLOPS]:  the actual count of the (useful) floating-point
            operations performed.  An estimate, Info [UMFPACK_FLOPS_ESTIMATE],
            was computed by umfpack_*_*symbolic.  The estimate is guaranteed to
            be an upper bound on this flop count.  The flop count excludes
            "useless" flops on zero values, flops performed during the pivot
            search (for tentative updates and assembly of candidate columns),
            and flops performed to add frontal matrices together.

            For the real version, only (+ - * /) are counted.  For the complex
            version, the following counts are used:

                operation       flops
                c = 1/b         6
                c = a*b         6
                c -= a*b        8

        Info [UMFPACK_LNZ]: the actual nonzero entries in final factor L,
            including the diagonal.  This excludes any zero entries in L,
            although some of these are stored in the Numeric object.  The
            Info [UMFPACK_LU_ENTRIES] statistic does account for all
            explicitly stored zeros, however.  Info [UMFPACK_LNZ_ESTIMATE],
            an estimate, was computed by umfpack_*_*symbolic.  The estimate is
            guaranteed to be an upper bound on Info [UMFPACK_LNZ].

        Info [UMFPACK_UNZ]: the actual nonzero entries in final factor U,
            including the diagonal.  This excludes any zero entries in U,
            although some of these are stored in the Numeric object.  The
            Info [UMFPACK_LU_ENTRIES] statistic does account for all
            explicitly stored zeros, however.  Info [UMFPACK_UNZ_ESTIMATE],
            an estimate, was computed by umfpack_*_*symbolic.  The estimate is
            guaranteed to be an upper bound on Info [UMFPACK_UNZ].

        Info [UMFPACK_NUMERIC_DEFRAG]:  The number of garbage collections
            performed during umfpack_*_numeric, to compact the contents of the
            variable-sized workspace used by umfpack_*_numeric.  No estimate was
            computed by umfpack_*_*symbolic.  In the current version of UMFPACK,
            garbage collection is performed and then the memory is reallocated,
            so this statistic is the same as Info [UMFPACK_NUMERIC_REALLOC],
            below.  It may differ in future releases.

        Info [UMFPACK_NUMERIC_REALLOC]:  The number of times that the Numeric
            object was increased in size from its initial size.  A rough upper
            bound on the peak size of the Numeric object was computed by
            umfpack_*_*symbolic, so reallocations should be rare.  However, if
            umfpack_*_numeric is unable to allocate that much storage, it
            reduces its request until either the allocation succeeds, or until
            it gets too small to do anything with.  If the memory that it
            finally got was small, but usable, then the reallocation count
            could be high.  No estimate of this count was computed by
            umfpack_*_*symbolic.

        Info [UMFPACK_NUMERIC_COSTLY_REALLOC]:  The number of times that the
            system realloc library routine (or mxRealloc for the mexFunction)
            had to move the workspace.  Realloc can sometimes increase the size
            of a block of memory without moving it, which is much faster.  This
            statistic will always be <= Info [UMFPACK_NUMERIC_REALLOC].  If your
            memory space is fragmented, then the number of "costly" realloc's
            will be equal to Info [UMFPACK_NUMERIC_REALLOC].

        Info [UMFPACK_COMPRESSED_PATTERN]:  The number of integers used to
            represent the pattern of L and U.

        Info [UMFPACK_LU_ENTRIES]:  The total number of numerical values that
            are stored for the LU factors.  Some of the values may be explicitly
            zero in order to save space (allowing for a smaller compressed
            pattern).

        Info [UMFPACK_NUMERIC_TIME]:  The CPU time taken, in seconds.

        Info [UMFPACK_RCOND]:  A rough estimate of the condition number, equal
            to min (abs (diag (U))) / max (abs (diag (U))), or zero if the
            diagonal of U is all zero.

        Info [UMFPACK_UDIAG_NZ]:  The number of numerically nonzero values on
            the diagonal of U.

        Info [UMFPACK_UMIN]:  the smallest absolute value on the diagonal of U.

        Info [UMFPACK_UMAX]:  the smallest absolute value on the diagonal of U.

        Info [UMFPACK_MAX_FRONT_SIZE]: the size of the
            largest frontal matrix (number of entries).

        Info [UMFPACK_NUMERIC_WALLTIME]:  The wallclock time taken, in seconds.

        Info [UMFPACK_MAX_FRONT_NROWS]: the max number of
            rows in any frontal matrix.

        Info [UMFPACK_MAX_FRONT_NCOLS]: the max number of
            columns in any frontal matrix.

        Info [UMFPACK_WAS_SCALED]:  the scaling used, either UMFPACK_SCALE_NONE,
            UMFPACK_SCALE_SUM, or UMFPACK_SCALE_MAX.

        Info [UMFPACK_RSMIN]: if scaling is performed, the smallest scale factor
            for any row (either the smallest sum of absolute entries, or the
            smallest maximum of absolute entries).

        Info [UMFPACK_RSMAX]: if scaling is performed, the largest scale factor
            for any row (either the largest sum of absolute entries, or the
            largest maximum of absolute entries).

        Info [UMFPACK_ALLOC_INIT_USED]:  the initial allocation parameter used.

        Info [UMFPACK_FORCED_UPDATES]:  the number of BLAS-3 updates to the
            frontal matrices that were required because the frontal matrix
            grew larger than its current working array.

        Info [UMFPACK_NOFF_DIAG]: number of off-diagonal pivots selected, if the
            symmetric strategy is used.

        Info [UMFPACK_NZDROPPED]: the number of entries smaller in absolute
            value than Control [UMFPACK_DROPTOL] that were dropped from L and U.
            Note that entries on the diagonal of U are never dropped.

        Info [UMFPACK_ALL_LNZ]: the number of entries in L, including the
            diagonal, if no small entries are dropped.

        Info [UMFPACK_ALL_UNZ]: the number of entries in U, including the
            diagonal, if no small entries are dropped.

        Only the above listed Info [...] entries are accessed.  The remaining
        entries of Info are not accessed or modified by umfpack_*_numeric.
        Future versions might modify different parts of Info.
*/

//------------------------------------------------------------------------------
// umfpack_solve
//------------------------------------------------------------------------------

int umfpack_di_solve
(
    int sys,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    double X [ ],
    const double B [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_dl_solve
(
    int sys,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    double X [ ],
    const double B [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_solve
(
    int sys,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    double Xx [ ],       double Xz [ ],
    const double Bx [ ], const double Bz [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zl_solve
(
    int sys,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    double Xx [ ],       double Xz [ ],
    const double Bx [ ], const double Bz [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t *Ap, *Ai ;
    int sys ;
    double *B, *X, *Ax, Info [UMFPACK_INFO], Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_solve (sys, Ap, Ai, Ax, X, B, Numeric, Control,
        Info) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t *Ap, *Ai ;
    int sys ;
    double *B, *X, *Ax, Info [UMFPACK_INFO], Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_solve (sys, Ap, Ai, Ax, X, B, Numeric, Control,
        Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t *Ap, *Ai ;
    int sys ;
    double *Bx, *Bz, *Xx, *Xz, *Ax, *Az, Info [UMFPACK_INFO],
        Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_solve (sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz,
        Numeric, Control, Info) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t *Ap, *Ai ;
    int sys ;
    double *Bx, *Bz, *Xx, *Xz, *Ax, *Az, Info [UMFPACK_INFO],
        Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_solve (sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz,
        Numeric, Control, Info) ;

packed complex Syntax:

    Same as above, Xz, Bz, and Az are NULL.

Purpose:

    Given LU factors computed by umfpack_*_numeric (PAQ=LU, PRAQ=LU, or
    P(R\A)Q=LU) and the right-hand-side, B, solve a linear system for the
    solution X.  Iterative refinement is optionally performed.  Only square
    systems are handled.  Singular matrices result in a divide-by-zero for all
    systems except those involving just the matrix L.  Iterative refinement is
    not performed for singular matrices.  In the discussion below, n is equal
    to n_row and n_col, because only square systems are handled.

Returns:

    The status code is returned.  See Info [UMFPACK_STATUS], below.

Arguments:

    int sys ;           Input argument, not modified.

        Defines which system to solve.  (') is the linear algebraic transpose
        (complex conjugate if A is complex), and (.') is the array transpose.

            sys value       system solved
            UMFPACK_A       Ax=b
            UMFPACK_At      A'x=b
            UMFPACK_Aat     A.'x=b
            UMFPACK_Pt_L    P'Lx=b
            UMFPACK_L       Lx=b
            UMFPACK_Lt_P    L'Px=b
            UMFPACK_Lat_P   L.'Px=b
            UMFPACK_Lt      L'x=b
            UMFPACK_U_Qt    UQ'x=b
            UMFPACK_U       Ux=b
            UMFPACK_Q_Ut    QU'x=b
            UMFPACK_Q_Uat   QU.'x=b
            UMFPACK_Ut      U'x=b
            UMFPACK_Uat     U.'x=b

        Iterative refinement can be optionally performed when sys is any of
        the following:

            UMFPACK_A       Ax=b
            UMFPACK_At      A'x=b
            UMFPACK_Aat     A.'x=b

        For the other values of the sys argument, iterative refinement is not
        performed (Control [UMFPACK_IRSTEP], Ap, Ai, Ax, and Az are ignored).

    Int Ap [n+1] ;      Input argument, not modified.
    Int Ai [nz] ;       Input argument, not modified.
    double Ax [nz] ;    Input argument, not modified.
                        Size 2*nz for packed complex case.
    double Az [nz] ;    Input argument, not modified, for complex versions.

        If iterative refinement is requested (Control [UMFPACK_IRSTEP] >= 1,
        Ax=b, A'x=b, or A.'x=b is being solved, and A is nonsingular), then
        these arrays must be identical to the same ones passed to
        umfpack_*_numeric.  The umfpack_*_solve routine does not check the
        contents of these arguments, so the results are undefined if Ap, Ai, Ax,
        and/or Az are modified between the calls the umfpack_*_numeric and
        umfpack_*_solve.  These three arrays do not need to be present (NULL
        pointers can be passed) if Control [UMFPACK_IRSTEP] is zero, or if a
        system other than Ax=b, A'x=b, or A.'x=b is being solved, or if A is
        singular, since in each of these cases A is not accessed.

        If Az, Xz, or Bz are NULL, then both real
        and imaginary parts are contained in Ax[0..2*nz-1], with Ax[2*k]
        and Ax[2*k+1] being the real and imaginary part of the kth entry.

    double X [n] ;      Output argument.
    or:
    double Xx [n] ;     Output argument, real part
                        Size 2*n for packed complex case.
    double Xz [n] ;     Output argument, imaginary part.

        The solution to the linear system, where n = n_row = n_col is the
        dimension of the matrices A, L, and U.

        If Az, Xz, or Bz are NULL, then both real
        and imaginary parts are returned in Xx[0..2*n-1], with Xx[2*k] and
        Xx[2*k+1] being the real and imaginary part of the kth entry.

    double B [n] ;      Input argument, not modified.
    or:
    double Bx [n] ;     Input argument, not modified, real part.
                        Size 2*n for packed complex case.
    double Bz [n] ;     Input argument, not modified, imaginary part.

        The right-hand side vector, b, stored as a conventional array of size n
        (or two arrays of size n for complex versions).  This routine does not
        solve for multiple right-hand-sides, nor does it allow b to be stored in
        a sparse-column form.

        If Az, Xz, or Bz are NULL, then both real
        and imaginary parts are contained in Bx[0..2*n-1], with Bx[2*k]
        and Bx[2*k+1] being the real and imaginary part of the kth entry.

    void *Numeric ;             Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_IRSTEP]:  The maximum number of iterative refinement
            steps to attempt.  A value less than zero is treated as zero.  If
            less than 1, or if Ax=b, A'x=b, or A.'x=b is not being solved, or
            if A is singular, then the Ap, Ai, Ax, and Az arguments are not
            accessed.  Default: 2.

    double Info [UMFPACK_INFO] ;        Output argument.

        Contains statistics about the solution factorization.  If a
        (double *) NULL pointer is passed, then no statistics are returned in
        Info (this is not an error condition).  The following statistics are
        computed in umfpack_*_solve:

        Info [UMFPACK_STATUS]: status code.  This is also the return value,
            whether or not Info is present.

            UMFPACK_OK

                The linear system was successfully solved.

            UMFPACK_WARNING_singular_matrix

                A divide-by-zero occurred.  Your solution will contain Inf's
                and/or NaN's.  Some parts of the solution may be valid.  For
                example, solving Ax=b with

                A = [2 0]  b = [ 1 ]  returns x = [ 0.5 ]
                    [0 0]      [ 0 ]              [ Inf ]

            UMFPACK_ERROR_out_of_memory

                Insufficient memory to solve the linear system.

            UMFPACK_ERROR_argument_missing

                One or more required arguments are missing.  The B, X, (or
                Bx and Xx for the complex versions) arguments
                are always required.  Info and Control are not required.  Ap,
                Ai, Ax are required if Ax=b,
                A'x=b, A.'x=b is to be solved, the (default) iterative
                refinement is requested, and the matrix A is nonsingular.

            UMFPACK_ERROR_invalid_system

                The sys argument is not valid, or the matrix A is not square.

            UMFPACK_ERROR_invalid_Numeric_object

                The Numeric object is not valid.

        Info [UMFPACK_NROW], Info [UMFPACK_NCOL]:
                The dimensions of the matrix A (L is n_row-by-n_inner and
                U is n_inner-by-n_col, with n_inner = min(n_row,n_col)).

        Info [UMFPACK_NZ]:  the number of entries in the input matrix, Ap [n],
            if iterative refinement is requested (Ax=b, A'x=b, or A.'x=b is
            being solved, Control [UMFPACK_IRSTEP] >= 1, and A is nonsingular).

        Info [UMFPACK_IR_TAKEN]:  The number of iterative refinement steps
            effectively taken.  The number of steps attempted may be one more
            than this; the refinement algorithm backtracks if the last
            refinement step worsens the solution.

        Info [UMFPACK_IR_ATTEMPTED]:   The number of iterative refinement steps
            attempted.  The number of times a linear system was solved is one
            more than this (once for the initial Ax=b, and once for each Ay=r
            solved for each iterative refinement step attempted).

        Info [UMFPACK_OMEGA1]:  sparse backward error estimate, omega1, if
            iterative refinement was performed, or -1 if iterative refinement
            not performed.

        Info [UMFPACK_OMEGA2]:  sparse backward error estimate, omega2, if
            iterative refinement was performed, or -1 if iterative refinement
            not performed.

        Info [UMFPACK_SOLVE_FLOPS]:  the number of floating point operations
            performed to solve the linear system.  This includes the work
            taken for all iterative refinement steps, including the backtrack
            (if any).

        Info [UMFPACK_SOLVE_TIME]:  The time taken, in seconds.

        Info [UMFPACK_SOLVE_WALLTIME]:  The wallclock time taken, in seconds.

        Only the above listed Info [...] entries are accessed.  The remaining
        entries of Info are not accessed or modified by umfpack_*_solve.
        Future versions might modify different parts of Info.
*/

//------------------------------------------------------------------------------
// umfpack_free_symbolic
//------------------------------------------------------------------------------

void umfpack_di_free_symbolic
(
    void **Symbolic
) ;

void umfpack_dl_free_symbolic
(
    void **Symbolic
) ;

void umfpack_zi_free_symbolic
(
    void **Symbolic
) ;

void umfpack_zl_free_symbolic
(
    void **Symbolic
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    umfpack_di_free_symbolic (&Symbolic) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    umfpack_dl_free_symbolic (&Symbolic) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    umfpack_zi_free_symbolic (&Symbolic) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    umfpack_zl_free_symbolic (&Symbolic) ;

Purpose:

    Deallocates the Symbolic object and sets the Symbolic handle to NULL.  This
    routine is the only valid way of destroying the Symbolic object.

Arguments:

    void **Symbolic ;       Input argument, set to (void *) NULL on output.

        Points to a valid Symbolic object computed by umfpack_*_symbolic.
        No action is taken if Symbolic is a (void *) NULL pointer.
*/

//------------------------------------------------------------------------------
// umfpack_free_numeric
//------------------------------------------------------------------------------

void umfpack_di_free_numeric
(
    void **Numeric
) ;

void umfpack_dl_free_numeric
(
    void **Numeric
) ;

void umfpack_zi_free_numeric
(
    void **Numeric
) ;

void umfpack_zl_free_numeric
(
    void **Numeric
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    umfpack_di_free_numeric (&Numeric) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    umfpack_dl_free_numeric (&Numeric) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    umfpack_zi_free_numeric (&Numeric) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    umfpack_zl_free_numeric (&Numeric) ;

Purpose:

    Deallocates the Numeric object and sets the Numeric handle to NULL.  This
    routine is the only valid way of destroying the Numeric object.

Arguments:

    void **Numeric ;        Input argument, set to (void *) NULL on output.

        Numeric points to a valid Numeric object, computed by umfpack_*_numeric.
        No action is taken if Numeric is a (void *) NULL pointer.
*/

//==============================================================================
//==== Alternative routines ====================================================
//==============================================================================

//------------------------------------------------------------------------------
// umfpack_defaults
//------------------------------------------------------------------------------

void umfpack_di_defaults
(
    double Control [UMFPACK_CONTROL]
) ;

void umfpack_dl_defaults
(
    double Control [UMFPACK_CONTROL]
) ;

void umfpack_zi_defaults
(
    double Control [UMFPACK_CONTROL]
) ;

void umfpack_zl_defaults
(
    double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_di_defaults (Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_dl_defaults (Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_zi_defaults (Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_zl_defaults (Control) ;

Purpose:

    Sets the default control parameter settings.

Arguments:

    double Control [UMFPACK_CONTROL] ;  Output argument.

        Control is set to the default control parameter settings.  You can
        then modify individual settings by changing specific entries in the
        Control array.  If Control is a (double *) NULL pointer, then
        umfpack_*_defaults returns silently (no error is generated, since
        passing a NULL pointer for Control to any UMFPACK routine is valid).
*/

//------------------------------------------------------------------------------
// umfpack_qsymbolic
//------------------------------------------------------------------------------

int umfpack_di_qsymbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    const int32_t Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_dl_qsymbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    const int64_t Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_qsymbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int32_t Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zl_qsymbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int64_t Qinit [ ],
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_di_fsymbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    int (*user_ordering) (int32_t, int32_t, int32_t, int32_t *, int32_t *,
        int32_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_dl_fsymbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    int (*user_ordering) (int64_t, int64_t, int64_t, int64_t *, int64_t *,
        int64_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_fsymbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    int (*user_ordering) (int32_t, int32_t, int32_t, int32_t *, int32_t *,
        int32_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zl_fsymbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    int (*user_ordering) (int64_t, int64_t, int64_t, int64_t *, int64_t *,
        int64_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int32_t n_row, n_col, *Ap, *Ai, *Qinit ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax ;
    int status = umfpack_di_qsymbolic (n_row, n_col, Ap, Ai, Ax, Qinit,
        &Symbolic, Control, Info) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int64_t n_row, n_col, *Ap, *Ai, *Qinit ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax ;
    int status = umfpack_dl_qsymbolic (n_row, n_col, Ap, Ai, Ax, Qinit,
        &Symbolic, Control, Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int32_t n_row, n_col, *Ap, *Ai, *Qinit ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax, *Az ;
    int status = umfpack_zi_qsymbolic (n_row, n_col, Ap, Ai, Ax, Az, Qinit,
        &Symbolic, Control, Info) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    int64_t n_row, n_col, *Ap, *Ai, *Qinit ;
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO], *Ax, *Az ;
    int status = umfpack_zl_qsymbolic (n_row, n_col, Ap, Ai, Ax, Az, Qinit,
        &Symbolic, Control, Info) ;

packed complex Syntax:

    Same as above, except Az is NULL.

Purpose:

    Given the nonzero pattern of a sparse matrix A in column-oriented form, and
    a sparsity preserving column pre-ordering Qinit, umfpack_*_qsymbolic
    performs the symbolic factorization of A*Qinit (or A (:,Qinit) in MATLAB
    notation).  This is identical to umfpack_*_symbolic, except that neither
    COLAMD nor AMD are called and the user input column order Qinit is used
    instead.  Note that in general, the Qinit passed to umfpack_*_qsymbolic
    can differ from the final Q found in umfpack_*_numeric.  The unsymmetric
    strategy will perform a column etree postordering done in
    umfpack_*_qsymbolic and sparsity-preserving modifications are made within
    each frontal matrix during umfpack_*_numeric.  The symmetric
    strategy will preserve Qinit, unless the matrix is structurally singular.

    See umfpack_*_symbolic for more information.  Note that Ax and Ax are
    optional.  The may be NULL.

    *** WARNING ***  A poor choice of Qinit can easily cause umfpack_*_numeric
    to use a huge amount of memory and do a lot of work.  The "default" symbolic
    analysis method is umfpack_*_symbolic, not this routine.  If you use this
    routine, the performance of UMFPACK is your responsibility;  UMFPACK will
    not try to second-guess a poor choice of Qinit.

Returns:

    The value of Info [UMFPACK_STATUS]; see umfpack_*_symbolic.
    Also returns UMFPACK_ERROR_invalid_permuation if Qinit is not a valid
    permutation vector.

Arguments:

    All arguments are the same as umfpack_*_symbolic, except for the following:

    Int Qinit [n_col] ;         Input argument, not modified.

        The user's fill-reducing initial column pre-ordering.  This must be a
        permutation of 0..n_col-1.  If Qinit [k] = j, then column j is the kth
        column of the matrix A (:,Qinit) to be factorized.  If Qinit is an
        (Int *) NULL pointer, then COLAMD or AMD are called instead.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If Qinit is not NULL, then only two strategies are recognized:
        the unsymmetric strategy and the symmetric strategy.
        If Control [UMFPACK_STRATEGY] is UMFPACK_STRATEGY_SYMMETRIC,
        then the symmetric strategy is used.  Otherwise the unsymmetric
        strategy is used.

The umfpack_*_fsymbolic functions are identical to their umfpack_*_qsymbolic
functions, except that Qinit is replaced with a pointer to an ordering
function (user_ordering), and a pointer to an extra input that is passed
to the user_ordering (user_params).  The arguments have the following syntax
(where Int is int32_t or int64_t):

    int (*user_ordering)    // TRUE if OK, FALSE otherwise
    (
        // inputs, not modified on output
        Int,            // nrow
        Int,            // ncol
        Int,            // sym: if TRUE and nrow==ncol do A+A', else do A'A
        Int *,          // Ap, size ncol+1
        Int *,          // Ai, size nz
        // output
        Int *,          // size ncol, fill-reducing permutation
        // input/output
        void *,         // user_params (ignored by UMFPACK)
        double *        // user_info[0..2], optional output for symmetric case.
                        // user_info[0]: max column count for L=chol(A+A')
                        // user_info[1]: nnz (L)
                        // user_info[2]: flop count for chol(A+A'), if A real
    ),
    void *user_params,  // passed to user_ordering function

*/

//------------------------------------------------------------------------------
// umfpack_paru: support functions for ParU
//------------------------------------------------------------------------------

int umfpack_di_paru_symbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    const int32_t Qinit [ ],
    int (*user_ordering) (int32_t, int32_t, int32_t, int32_t *, int32_t *,
        int32_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_dl_paru_symbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    const int64_t Qinit [ ],
    int (*user_ordering) (int64_t, int64_t, int64_t, int64_t *, int64_t *,
        int64_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zi_paru_symbolic
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int32_t Qinit [ ],
    int (*user_ordering) (int32_t, int32_t, int32_t, int32_t *, int32_t *,
        int32_t *, void *, double *),
    void *user_params,
    void **Symbolic,
    void **SW,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO]
) ;

int umfpack_zl_paru_symbolic
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int64_t Qinit [ ],
    int (*user_ordering) (int64_t, int64_t, int64_t, int64_t *, int64_t *,
        int64_t *, void *, double *),
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


//------------------------------------------------------------------------------
// umfpack_wsolve
//------------------------------------------------------------------------------

int umfpack_di_wsolve
(
    int sys,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    double X [ ],
    const double B [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO],
    int32_t Wi [ ],
    double W [ ]
) ;

int umfpack_dl_wsolve
(
    int sys,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    double X [ ],
    const double B [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO],
    int64_t Wi [ ],
    double W [ ]
) ;

int umfpack_zi_wsolve
(
    int32_t sys,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    double Xx [ ],       double Xz [ ],
    const double Bx [ ], const double Bz [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO],
    int32_t Wi [ ],
    double W [ ]
) ;

int umfpack_zl_wsolve
(
    int sys,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    double Xx [ ],       double Xz [ ],
    const double Bx [ ], const double Bz [ ],
    void *Numeric,
    const double Control [UMFPACK_CONTROL],
    double Info [UMFPACK_INFO],
    int64_t Wi [ ],
    double W [ ]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t *Ap, *Ai, *Wi ;
    int sys ;
    double *B, *X, *Ax, *W, Info [UMFPACK_INFO], Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_wsolve (sys, Ap, Ai, Ax, X, B, Numeric,
        Control, Info, Wi, W) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t *Ap, *Ai, *Wi ;
    int sys ;
    double *B, *X, *Ax, *W, Info [UMFPACK_INFO], Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_wsolve (sys, Ap, Ai, Ax, X, B, Numeric,
        Control, Info, Wi, W) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t *Ap, *Ai, *Wi ; 
    int sys ;
    double *Bx, *Bz, *Xx, *Xz, *Ax, *Az, *W,
        Info [UMFPACK_INFO], Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_wsolve (sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz,
        Numeric, Control, Info, Wi, W) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t *Ap, *Ai, *Wi ;
    int sys ;
    double *Bx, *Bz, *Xx, *Xz, *Ax, *Az, *W,
        Info [UMFPACK_INFO], Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_wsolve (sys, Ap, Ai, Ax, Az, Xx, Xz, Bx, Bz,
        Numeric, Control, Info, Wi, W) ;

packed complex Syntax:

    Same as above, except Az, Xz, and Bz are NULL.

Purpose:

    Given LU factors computed by umfpack_*_numeric (PAQ=LU) and the
    right-hand-side, B, solve a linear system for the solution X.  Iterative
    refinement is optionally performed.  This routine is identical to
    umfpack_*_solve, except that it does not dynamically allocate any workspace.
    When you have many linear systems to solve, this routine is faster than
    umfpack_*_solve, since the workspace (Wi, W) needs to be allocated only
    once, prior to calling umfpack_*_wsolve.

Returns:

    The status code is returned.  See Info [UMFPACK_STATUS], below.

Arguments:

    Int sys ;           Input argument, not modified.
    Int Ap [n+1] ;      Input argument, not modified.
    Int Ai [nz] ;       Input argument, not modified.
    double Ax [nz] ;    Input argument, not modified.
                        Size 2*nz in packed complex case.
    double X [n] ;      Output argument.
    double B [n] ;      Input argument, not modified.
    void *Numeric ;     Input argument, not modified.
    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.
    double Info [UMFPACK_INFO] ;        Output argument.

    for complex versions:
    double Az [nz] ;    Input argument, not modified, imaginary part
    double Xx [n] ;     Output argument, real part.
                        Size 2*n in packed complex case.
    double Xz [n] ;     Output argument, imaginary part
    double Bx [n] ;     Input argument, not modified, real part.
                        Size 2*n in packed complex case.
    double Bz [n] ;     Input argument, not modified, imaginary part

        The above arguments are identical to umfpack_*_solve, except that the
        error code UMFPACK_ERROR_out_of_memory will not be returned in
        Info [UMFPACK_STATUS], since umfpack_*_wsolve does not allocate any
        memory.

    Int Wi [n] ;                Workspace.
    double W [c*n] ;            Workspace, where c is defined below.

        The Wi and W arguments are workspace used by umfpack_*_wsolve.  They
        need not be initialized on input, and their contents are undefined on
        output.  The size of W depends on whether or not iterative refinement is
        used, and which version (real or complex) is called.  Iterative
        refinement is performed if Ax=b, A'x=b, or A.'x=b is being solved,
        Control [UMFPACK_IRSTEP] > 0, and A is nonsingular.  The size of W is
        given below:

                                no iter.        with iter.
                                refinement      refinement
        umfpack_di_wsolve       n               5*n
        umfpack_dl_wsolve       n               5*n
        umfpack_zi_wsolve       4*n             10*n
        umfpack_zl_wsolve       4*n             10*n
*/

//==============================================================================
//==== Matrix manipulation routines ============================================
//==============================================================================

//------------------------------------------------------------------------------
// umfpack_triplet_to_col
//------------------------------------------------------------------------------

int umfpack_di_triplet_to_col
(
    int32_t n_row,
    int32_t n_col,
    int32_t nz,
    const int32_t Ti [ ],
    const int32_t Tj [ ],
    const double Tx [ ],
    int32_t Ap [ ],
    int32_t Ai [ ],
    double Ax [ ],
    int32_t Map [ ]
) ;

int umfpack_dl_triplet_to_col
(
    int64_t n_row,
    int64_t n_col,
    int64_t nz,
    const int64_t Ti [ ],
    const int64_t Tj [ ],
    const double Tx [ ],
    int64_t Ap [ ],
    int64_t Ai [ ],
    double Ax [ ],
    int64_t Map [ ]
) ;

int umfpack_zi_triplet_to_col
(
    int32_t n_row,
    int32_t n_col,
    int32_t nz,
    const int32_t Ti [ ],
    const int32_t Tj [ ],
    const double Tx [ ], const double Tz [ ],
    int32_t Ap [ ],
    int32_t Ai [ ],
    double Ax [ ], double Az [ ],
    int32_t Map [ ]
) ;

int umfpack_zl_triplet_to_col
(
    int64_t n_row,
    int64_t n_col,
    int64_t nz,
    const int64_t Ti [ ],
    const int64_t Tj [ ],
    const double Tx [ ], const double Tz [ ],
    int64_t Ap [ ],
    int64_t Ai [ ],
    double Ax [ ], double Az [ ],
    int64_t Map [ ]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, nz, *Ti, *Tj, *Ap, *Ai, *Map ;
    double *Tx, *Ax ;
    int status = umfpack_di_triplet_to_col (n_row, n_col, nz, Ti, Tj, Tx,
        Ap, Ai, Ax, Map) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, nz, *Ti, *Tj, *Ap, *Ai, *Map ;
    double *Tx, *Ax ;
    int status = umfpack_dl_triplet_to_col (n_row, n_col, nz, Ti, Tj, Tx,
        Ap, Ai, Ax, Map) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, nz, *Ti, *Tj, *Ap, *Ai, *Map ;
    double *Tx, *Tz, *Ax, *Az ;
    int status = umfpack_zi_triplet_to_col (n_row, n_col, nz, Ti, Tj, Tx, Tz,
        Ap, Ai, Ax, Az, Map) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, nz, *Ti, *Tj, *Ap, *Ai, *Map ;
    double *Tx, *Tz, *Ax, *Az ;
    int status = umfpack_zl_triplet_to_col (n_row, n_col, nz, Ti, Tj, Tx, Tz,
        Ap, Ai, Ax, Az, Map) ;

packed complex Syntax:

    Same as above, except Tz and Az are NULL.

Purpose:

    Converts a sparse matrix from "triplet" form to compressed-column form.
    Analogous to A = spconvert (Ti, Tj, Tx + Tz*1i) in MATLAB, except that
    zero entries present in the triplet form are present in A.

    The triplet form of a matrix is a very simple data structure for basic
    sparse matrix operations.  For example, suppose you wish to factorize a
    matrix A coming from a finite element method, in which A is a sum of
    dense submatrices, A = E1 + E2 + E3 + ... .  The entries in each element
    matrix Ei can be concatenated together in the three triplet arrays, and
    any overlap between the elements will be correctly summed by
    umfpack_*_triplet_to_col.

    Transposing a matrix in triplet form is simple; just interchange the
    use of Ti and Tj.  You can construct the complex conjugate transpose by
    negating Tz, for the complex versions.

    Permuting a matrix in triplet form is also simple.  If you want the matrix
    PAQ, or A (P,Q) in MATLAB notation, where P [k] = i means that row i of
    A is the kth row of PAQ and Q [k] = j means that column j of A is the kth
    column of PAQ, then do the following.  First, create inverse permutations
    Pinv and Qinv such that Pinv [i] = k if P [k] = i and Qinv [j] = k if
    Q [k] = j.  Next, for the mth triplet (Ti [m], Tj [m], Tx [m], Tz [m]),
    replace Ti [m] with Pinv [Ti [m]] and replace Tj [m] with Qinv [Tj [m]].

    If you have a column-form matrix with duplicate entries or unsorted
    columns, you can sort it and sum up the duplicates by first converting it
    to triplet form with umfpack_*_col_to_triplet, and then converting it back
    with umfpack_*_triplet_to_col.

    Constructing a submatrix is also easy.  Just scan the triplets and remove
    those entries outside the desired subset of 0...n_row-1 and 0...n_col-1,
    and renumber the indices according to their position in the subset.

    You can do all these operations on a column-form matrix by first
    converting it to triplet form with umfpack_*_col_to_triplet, doing the
    operation on the triplet form, and then converting it back with
    umfpack_*_triplet_to_col.

    The only operation not supported easily in the triplet form is the
    multiplication of two sparse matrices (UMFPACK does not provide this
    operation).

    You can print the input triplet form with umfpack_*_report_triplet, and
    the output matrix with umfpack_*_report_matrix.

    The matrix may be singular (nz can be zero, and empty rows and/or columns
    may exist).  It may also be rectangular and/or complex.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_argument_missing if Ap, Ai, Ti, and/or Tj are missing.
    UMFPACK_ERROR_n_nonpositive if n_row <= 0 or n_col <= 0.
    UMFPACK_ERROR_invalid_matrix if nz < 0, or if for any k, Ti [k] and/or
        Tj [k] are not in the range 0 to n_row-1 or 0 to n_col-1, respectively.
    UMFPACK_ERROR_out_of_memory if unable to allocate sufficient workspace.

Arguments:

    Int n_row ;         Input argument, not modified.
    Int n_col ;         Input argument, not modified.

        A is an n_row-by-n_col matrix.  Restriction: n_row > 0 and n_col > 0.
        All row and column indices in the triplet form must be in the range
        0 to n_row-1 and 0 to n_col-1, respectively.

    Int nz ;            Input argument, not modified.

        The number of entries in the triplet form of the matrix.  Restriction:
        nz >= 0.

    Int Ti [nz] ;       Input argument, not modified.
    Int Tj [nz] ;       Input argument, not modified.
    double Tx [nz] ;    Input argument, not modified.
                        Size 2*nz if Tz or Az are NULL.
    double Tz [nz] ;    Input argument, not modified, for complex versions.

        Ti, Tj, Tx, and Tz hold the "triplet" form of a sparse matrix.  The kth
        nonzero entry is in row i = Ti [k], column j = Tj [k], and the real part
        of a_ij is Tx [k].  The imaginary part of a_ij is Tz [k], for complex
        versions.  The row and column indices i and j must be in the range 0 to
        n_row-1 and 0 to n_col-1, respectively.  Duplicate entries may be
        present; they are summed in the output matrix.  This is not an error
        condition.  The "triplets" may be in any order.  Tx, Tz, Ax, and Az
        are optional.  Ax is computed only if both Ax and Tx are present
        (not (double *) NULL).  This is not error condition; the routine can
        create just the pattern of the output matrix from the pattern of the
        triplets.

        If Az or Tz are NULL, then both real
        and imaginary parts are contained in Tx[0..2*nz-1], with Tx[2*k]
        and Tx[2*k+1] being the real and imaginary part of the kth entry.

    Int Ap [n_col+1] ;  Output argument.

        Ap is an integer array of size n_col+1 on input.  On output, Ap holds
        the "pointers" for the column form of the sparse matrix A.  Column j of
        the matrix A is held in Ai [(Ap [j]) ... (Ap [j+1]-1)].  The first
        entry, Ap [0], is zero, and Ap [j] <= Ap [j+1] holds for all j in the
        range 0 to n_col-1.  The value nz2 = Ap [n_col] is thus the total
        number of entries in the pattern of the matrix A.  Equivalently, the
        number of duplicate triplets is nz - Ap [n_col].

    Int Ai [nz] ;       Output argument.

        Ai is an integer array of size nz on input.  Note that only the first
        Ap [n_col] entries are used.

        The nonzero pattern (row indices) for column j is stored in
        Ai [(Ap [j]) ... (Ap [j+1]-1)].  The row indices in a given column j
        are in ascending order, and no duplicate row indices are present.
        Row indices are in the range 0 to n_col-1 (the matrix is 0-based).

    double Ax [nz] ;    Output argument.  Size 2*nz if Tz or Az are NULL.
    double Az [nz] ;    Output argument for complex versions.

        Ax and Az (for the complex versions) are double arrays of size nz on
        input.  Note that only the first Ap [n_col] entries are used
        in both arrays.

        Ax is optional; if Tx and/or Ax are not present (a (double *) NULL
        pointer), then Ax is not computed.  If present, Ax holds the
        numerical values of the the real part of the sparse matrix A and Az
        holds the imaginary parts.  The nonzero pattern (row indices) for
        column j is stored in Ai [(Ap [j]) ... (Ap [j+1]-1)], and the
        corresponding numerical values are stored in
        Ax [(Ap [j]) ... (Ap [j+1]-1)].  The imaginary parts are stored in
        Az [(Ap [j]) ... (Ap [j+1]-1)], for the complex versions.

        If Az or Tz are NULL, then both real
        and imaginary parts are returned in Ax[0..2*nz2-1], with Ax[2*k]
        and Ax[2*k+1] being the real and imaginary part of the kth entry.

    Int Map [nz] ;      Optional output argument.

        If Map is present (a non-NULL pointer to an Int array of size nz), then
        on output it holds the position of the triplets in the column-form
        matrix.  That is, suppose p = Map [k], and the k-th triplet is i=Ti[k],
        j=Tj[k], and aij=Tx[k].  Then i=Ai[p], and aij will have been summed
        into Ax[p] (or simply aij=Ax[p] if there were no duplicate entries also
        in row i and column j).  Also, Ap[j] <= p < Ap[j+1].  The Map array is
        not computed if it is (Int *) NULL.  The Map array is useful for
        converting a subsequent triplet form matrix with the same pattern as the
        first one, without calling this routine.  If Ti and Tj do not change,
        then Ap, and Ai can be reused from the prior call to
        umfpack_*_triplet_to_col.  You only need to recompute Ax (and Az for the
        split complex version).  This code excerpt properly sums up all
        duplicate values (for the real version):

            for (p = 0 ; p < Ap [n_col] ; p++) Ax [p] = 0 ;
            for (k = 0 ; k < nz ; k++) Ax [Map [k]] += Tx [k] ;

        This feature is useful (along with the reuse of the Symbolic object) if
        you need to factorize a sequence of triplet matrices with identical
        nonzero pattern (the order of the triplets in the Ti,Tj,Tx arrays must
        also remain unchanged).  It is faster than calling this routine for
        each matrix, and requires no workspace.
*/

//------------------------------------------------------------------------------
// umfpack_col_to_triplet
//------------------------------------------------------------------------------

int umfpack_di_col_to_triplet
(
    int32_t n_col,
    const int32_t Ap [ ],
    int32_t Tj [ ]
) ;

int umfpack_dl_col_to_triplet
(
    int64_t n_col,
    const int64_t Ap [ ],
    int64_t Tj [ ]
) ;

int umfpack_zi_col_to_triplet
(
    int32_t n_col,
    const int32_t Ap [ ],
    int32_t Tj [ ]
) ;

int umfpack_zl_col_to_triplet
(
    int64_t n_col,
    const int64_t Ap [ ],
    int64_t Tj [ ]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t n_col, *Tj, *Ap ;
    int status = umfpack_di_col_to_triplet (n_col, Ap, Tj) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n_col, *Tj, *Ap ;
    int status = umfpack_dl_col_to_triplet (n_col, Ap, Tj) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n_col, *Tj, *Ap ;
    int status = umfpack_zi_col_to_triplet (n_col, Ap, Tj) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n_col, *Tj, *Ap ;
    int status = umfpack_zl_col_to_triplet (n_col, Ap, Tj) ;

Purpose:

    Converts a column-oriented matrix to a triplet form.  Only the column
    pointers, Ap, are required, and only the column indices of the triplet form
    are constructed.   This routine is the opposite of umfpack_*_triplet_to_col.
    The matrix may be singular and/or rectangular.  Analogous to [i, Tj, x] =
    find (A) in MATLAB, except that zero entries present in the column-form of
    A are present in the output, and i and x are not created (those are just Ai
    and Ax+Az*1i, respectively, for a column-form matrix A).

Returns:

    UMFPACK_OK if successful
    UMFPACK_ERROR_argument_missing if Ap or Tj is missing
    UMFPACK_ERROR_n_nonpositive if n_col <= 0
    UMFPACK_ERROR_invalid_matrix if Ap [n_col] < 0, Ap [0] != 0, or
        Ap [j] > Ap [j+1] for any j in the range 0 to n-1.
    Unsorted columns and duplicate entries do not cause an error (these would
    only be evident by examining Ai).  Empty rows and columns are OK.

Arguments:

    Int n_col ;         Input argument, not modified.

        A is an n_row-by-n_col matrix.  Restriction: n_col > 0.
        (n_row is not required)

    Int Ap [n_col+1] ;  Input argument, not modified.

        The column pointers of the column-oriented form of the matrix.  See
        umfpack_*_*symbolic for a description.  The number of entries in
        the matrix is nz = Ap [n_col].  Restrictions on Ap are the same as those
        for umfpack_*_transpose.  Ap [0] must be zero, nz must be >= 0, and
        Ap [j] <= Ap [j+1] and Ap [j] <= Ap [n_col] must be true for all j in
        the range 0 to n_col-1.  Empty columns are OK (that is, Ap [j] may equal
        Ap [j+1] for any j in the range 0 to n_col-1).

    Int Tj [nz] ;       Output argument.

        Tj is an integer array of size nz on input, where nz = Ap [n_col].
        Suppose the column-form of the matrix is held in Ap, Ai, Ax, and Az
        (see umfpack_*_*symbolic for a description).  Then on output, the
        triplet form of the same matrix is held in Ai (row indices), Tj (column
        indices), and Ax (numerical values).  Note, however, that this routine
        does not require Ai and Ax (or Az for the complex version) in order to
        do the conversion.
*/

//------------------------------------------------------------------------------
// umfpack_transpose
//------------------------------------------------------------------------------

int umfpack_di_transpose
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    const int32_t P [ ],
    const int32_t Q [ ],
    int32_t Rp [ ],
    int32_t Ri [ ],
    double Rx [ ]
) ;

int umfpack_dl_transpose
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    const int64_t P [ ],
    const int64_t Q [ ],
    int64_t Rp [ ],
    int64_t Ri [ ],
    double Rx [ ]
) ;

int umfpack_zi_transpose
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int32_t P [ ],
    const int32_t Q [ ],
    int32_t Rp [ ],
    int32_t Ri [ ],
    double Rx [ ], double Rz [ ],
    int do_conjugate
) ;

int umfpack_zl_transpose
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    const int64_t P [ ],
    const int64_t Q [ ],
    int64_t Rp [ ],
    int64_t Ri [ ],
    double Rx [ ], double Rz [ ],
    int do_conjugate
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, *Ap, *Ai, *P, *Q, *Rp, *Ri ;
    double *Ax, *Rx ;
    int status = umfpack_di_transpose (n_row, n_col, Ap, Ai, Ax, P, Q,
        Rp, Ri, Rx) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, *Ap, *Ai, *P, *Q, *Rp, *Ri ;
    double *Ax, *Rx ;
    int status = umfpack_dl_transpose (n_row, n_col, Ap, Ai, Ax, P, Q,
        Rp, Ri, Rx) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, *Ap, *Ai, *P, *Q, *Rp, *Ri ;
    int do_conjugate ;
    double *Ax, *Az, *Rx, *Rz ;
    int status = umfpack_zi_transpose (n_row, n_col, Ap, Ai, Ax, Az, P, Q,
        Rp, Ri, Rx, Rz, do_conjugate) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, *Ap, *Ai, *P, *Q, *Rp, *Ri ;
    int do_conjugate ;
    double *Ax, *Az, *Rx, *Rz ;
    int status = umfpack_zl_transpose (n_row, n_col, Ap, Ai, Ax, Az, P, Q,
        Rp, Ri, Rx, Rz, do_conjugate) ;

packed complex Syntax:

    Same as above, except Az are Rz are NULL.

Purpose:

    Transposes and optionally permutes a sparse matrix in row or column-form,
    R = (PAQ)'.  In MATLAB notation, R = (A (P,Q))' or R = (A (P,Q)).' doing
    either the linear algebraic transpose or the array transpose. Alternatively,
    this routine can be viewed as converting A (P,Q) from column-form to
    row-form, or visa versa (for the array transpose).  Empty rows and columns
    may exist.  The matrix A may be singular and/or rectangular.

    umfpack_*_transpose is useful if you want to factorize A' or A.' instead of
    A.  Factorizing A' or A.' instead of A can be much better, particularly if
    AA' is much sparser than A'A.  You can still solve Ax=b if you factorize
    A' or A.', by solving with the sys argument UMFPACK_At or UMFPACK_Aat,
    respectively, in umfpack_*_*solve.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if umfpack_*_transpose fails to allocate a
        size-max (n_row,n_col) workspace.
    UMFPACK_ERROR_argument_missing if Ai, Ap, Ri, and/or Rp are missing.
    UMFPACK_ERROR_n_nonpositive if n_row <= 0 or n_col <= 0
    UMFPACK_ERROR_invalid_permutation if P and/or Q are invalid.
    UMFPACK_ERROR_invalid_matrix if Ap [n_col] < 0, if Ap [0] != 0,
        if Ap [j] > Ap [j+1] for any j in the range 0 to n_col-1,
        if any row index i is < 0 or >= n_row, or if the row indices
        in any column are not in ascending order.

Arguments:

    Int n_row ;         Input argument, not modified.
    Int n_col ;         Input argument, not modified.

        A is an n_row-by-n_col matrix.  Restriction: n_row > 0 and n_col > 0.

    Int Ap [n_col+1] ;  Input argument, not modified.

        The column pointers of the column-oriented form of the matrix A.  See
        umfpack_*_symbolic for a description.  The number of entries in
        the matrix is nz = Ap [n_col].  Ap [0] must be zero, Ap [n_col] must be
        => 0, and Ap [j] <= Ap [j+1] and Ap [j] <= Ap [n_col] must be true for
        all j in the range 0 to n_col-1.  Empty columns are OK (that is, Ap [j]
        may equal Ap [j+1] for any j in the range 0 to n_col-1).

    Int Ai [nz] ;       Input argument, not modified, of size nz = Ap [n_col].

        The nonzero pattern (row indices) for column j is stored in
        Ai [(Ap [j]) ... (Ap [j+1]-1)].  The row indices in a given column j
        must be in ascending order, and no duplicate row indices may be present.
        Row indices must be in the range 0 to n_row-1 (the matrix is 0-based).

    double Ax [nz] ;    Input argument, not modified, of size nz = Ap [n_col].
                        Size 2*nz if Az or Rz are NULL.
    double Az [nz] ;    Input argument, not modified, for complex versions.

        If present, these are the numerical values of the sparse matrix A.
        The nonzero pattern (row indices) for column j is stored in
        Ai [(Ap [j]) ... (Ap [j+1]-1)], and the corresponding real numerical
        values are stored in Ax [(Ap [j]) ... (Ap [j+1]-1)].  The imaginary
        values are stored in Az [(Ap [j]) ... (Ap [j+1]-1)].  The values are
        transposed only if Ax and Rx are present.
        This is not an error conditions; you are able to transpose
        and permute just the pattern of a matrix.

        If Az or Rz are NULL, then both real
        and imaginary parts are contained in Ax[0..2*nz-1], with Ax[2*k]
        and Ax[2*k+1] being the real and imaginary part of the kth entry.

    Int P [n_row] ;             Input argument, not modified.

        The permutation vector P is defined as P [k] = i, where the original
        row i of A is the kth row of PAQ.  If you want to use the identity
        permutation for P, simply pass (Int *) NULL for P.  This is not an error
        condition.  P is a complete permutation of all the rows of A; this
        routine does not support the creation of a transposed submatrix of A
        (R = A (1:3,:)' where A has more than 3 rows, for example, cannot be
        done; a future version might support this operation).

    Int Q [n_col] ;             Input argument, not modified.

        The permutation vector Q is defined as Q [k] = j, where the original
        column j of A is the kth column of PAQ.  If you want to use the identity
        permutation for Q, simply pass (Int *) NULL for Q.  This is not an error
        condition.  Q is a complete permutation of all the columns of A; this
        routine does not support the creation of a transposed submatrix of A.

    Int Rp [n_row+1] ;  Output argument.

        The column pointers of the matrix R = (A (P,Q))' or (A (P,Q)).', in the
        same form as the column pointers Ap for the matrix A.

    Int Ri [nz] ;       Output argument.

        The row indices of the matrix R = (A (P,Q))' or (A (P,Q)).' , in the
        same form as the row indices Ai for the matrix A.

    double Rx [nz] ;    Output argument.
                        Size 2*nz if Az or Rz are NULL.
    double Rz [nz] ;    Output argument, imaginary part for complex versions.

        If present, these are the numerical values of the sparse matrix R,
        in the same form as the values Ax and Az of the matrix A.

        If Az or Rz are NULL, then both real
        and imaginary parts are contained in Rx[0..2*nz-1], with Rx[2*k]
        and Rx[2*k+1] being the real and imaginary part of the kth entry.

    Int do_conjugate ;  Input argument for complex versions only.

        If true, and if Ax and Rx are present, then the linear
        algebraic transpose is computed (complex conjugate).  If false, the
        array transpose is computed instead.
*/

//------------------------------------------------------------------------------
// umfpack_scale
//------------------------------------------------------------------------------

int umfpack_di_scale
(
    double X [ ],
    const double B [ ],
    void *Numeric
) ;

int umfpack_dl_scale
(
    double X [ ],
    const double B [ ],
    void *Numeric
) ;

int umfpack_zi_scale
(
    double Xx [ ],       double Xz [ ],
    const double Bx [ ], const double Bz [ ],
    void *Numeric
) ;

int umfpack_zl_scale
(
    double Xx [ ],       double Xz [ ],
    const double Bx [ ], const double Bz [ ],
    void *Numeric
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double *B, *X ;
    int status = umfpack_di_scale (X, B, Numeric) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double *B, *X ;
    int status = umfpack_dl_scale (X, B, Numeric) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double *Bx, *Bz, *Xx, *Xz ;
    int status = umfpack_zi_scale (Xx, Xz, Bx, Bz, Numeric) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double *Bx, *Bz, *Xx, *Xz ;
    int status = umfpack_zl_scale (Xx, Xz, Bx, Bz, Numeric) ;

packed complex Syntax:

    Same as above, except both Xz and Bz are NULL.

Purpose:

    Given LU factors computed by umfpack_*_numeric (PAQ=LU, PRAQ=LU, or
    P(R\A)Q=LU), and a vector B, this routine computes X = B, X = R*B, or
    X = R\B, as appropriate.  X and B must be vectors equal in length to the
    number of rows of A.

Returns:

    The status code is returned.  UMFPACK_OK is returned if successful.
    UMFPACK_ERROR_invalid_Numeric_object is returned in the Numeric
    object is invalid.  UMFPACK_ERROR_argument_missing is returned if
    any of the input vectors are missing (X and B for the real version,
    and Xx and Bx for the complex version).

Arguments:

    double X [n_row] ;  Output argument.
    or:
    double Xx [n_row] ; Output argument, real part.
                        Size 2*n_row for packed complex case.
    double Xz [n_row] ; Output argument, imaginary part.

        The output vector X.  If either Xz or Bz are NULL, the vector
        X is in packed complex form, with the kth entry in Xx [2*k] and
        Xx [2*k+1], and likewise for B.

    double B [n_row] ;  Input argument, not modified.
    or:
    double Bx [n_row] ; Input argument, not modified, real part.
                        Size 2*n_row for packed complex case.
    double Bz [n_row] ; Input argument, not modified, imaginary part.

        The input vector B.  See above if either Xz or Bz are NULL.

    void *Numeric ;             Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric.

*/

//==============================================================================
//==== Getting the contents of the Symbolic and Numeric opaque objects =========
//==============================================================================

//------------------------------------------------------------------------------
// umfpack_get_lunz
//------------------------------------------------------------------------------

int umfpack_di_get_lunz
(
    int32_t *lnz,
    int32_t *unz,
    int32_t *n_row,
    int32_t *n_col,
    int32_t *nz_udiag,
    void *Numeric
) ;

int umfpack_dl_get_lunz
(
    int64_t *lnz,
    int64_t *unz,
    int64_t *n_row,
    int64_t *n_col,
    int64_t *nz_udiag,
    void *Numeric
) ;

int umfpack_zi_get_lunz
(
    int32_t *lnz,
    int32_t *unz,
    int32_t *n_row,
    int32_t *n_col,
    int32_t *nz_udiag,
    void *Numeric
) ;

int umfpack_zl_get_lunz
(
    int64_t *lnz,
    int64_t *unz,
    int64_t *n_row,
    int64_t *n_col,
    int64_t *nz_udiag,
    void *Numeric
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t lnz, unz, n_row, n_col, nz_udiag ;
    int status = umfpack_di_get_lunz (&lnz, &unz, &n_row, &n_col, &nz_udiag,
        Numeric) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t lnz, unz, n_row, n_col, nz_udiag ;
    int status = umfpack_dl_get_lunz (&lnz, &unz, &n_row, &n_col, &nz_udiag,
        Numeric) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t lnz, unz, n_row, n_col, nz_udiag ;
    int status = umfpack_zi_get_lunz (&lnz, &unz, &n_row, &n_col, &nz_udiag,
        Numeric) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t lnz, unz, n_row, n_col, nz_udiag ;
    int status = umfpack_zl_get_lunz (&lnz, &unz, &n_row, &n_col, &nz_udiag,
        Numeric) ;

Purpose:

    Determines the size and number of nonzeros in the LU factors held by the
    Numeric object.  These are also the sizes of the output arrays required
    by umfpack_*_get_numeric.

    The matrix L is n_row -by- min(n_row,n_col), with lnz nonzeros, including
    the entries on the unit diagonal of L.

    The matrix U is min(n_row,n_col) -by- n_col, with unz nonzeros, including
    nonzeros on the diagonal of U.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Numeric_object if Numeric is not a valid object.
    UMFPACK_ERROR_argument_missing if any other argument is (Int *) NULL.

Arguments:

    Int *lnz ;          Output argument.

        The number of nonzeros in L, including the diagonal (which is all
        one's).  This value is the required size of the Lj and Lx arrays as
        computed by umfpack_*_get_numeric.  The value of lnz is identical to
        Info [UMFPACK_LNZ], if that value was returned by umfpack_*_numeric.

    Int *unz ;          Output argument.

        The number of nonzeros in U, including the diagonal.  This value is the
        required size of the Ui and Ux arrays as computed by
        umfpack_*_get_numeric.  The value of unz is identical to
        Info [UMFPACK_UNZ], if that value was returned by umfpack_*_numeric.

    Int *n_row ;        Output argument.
    Int *n_col ;        Output argument.

        The order of the L and U matrices.  L is n_row -by- min(n_row,n_col)
        and U is min(n_row,n_col) -by- n_col.

    Int *nz_udiag ;     Output argument.

        The number of numerically nonzero values on the diagonal of U.  The
        matrix is singular if nz_diag < min(n_row,n_col).  A divide-by-zero
        will occur if nz_diag < n_row == n_col when solving a sparse system
        involving the matrix U in umfpack_*_*solve.  The value of nz_udiag is
        identical to Info [UMFPACK_UDIAG_NZ] if that value was returned by
        umfpack_*_numeric.

    void *Numeric ;     Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric.
*/

//------------------------------------------------------------------------------
// umfpack_get_numeric
//------------------------------------------------------------------------------

int umfpack_di_get_numeric
(
    int32_t Lp [ ],
    int32_t Lj [ ],
    double Lx [ ],
    int32_t Up [ ],
    int32_t Ui [ ],
    double Ux [ ],
    int32_t P [ ],
    int32_t Q [ ],
    double Dx [ ],
    int32_t *do_recip,
    double Rs [ ],
    void *Numeric
) ;

int umfpack_dl_get_numeric
(
    int64_t Lp [ ],
    int64_t Lj [ ],
    double Lx [ ],
    int64_t Up [ ],
    int64_t Ui [ ],
    double Ux [ ],
    int64_t P [ ],
    int64_t Q [ ],
    double Dx [ ],
    int64_t *do_recip,
    double Rs [ ],
    void *Numeric
) ;

int umfpack_zi_get_numeric
(
    int32_t Lp [ ],
    int32_t Lj [ ],
    double Lx [ ], double Lz [ ],
    int32_t Up [ ],
    int32_t Ui [ ],
    double Ux [ ], double Uz [ ],
    int32_t P [ ],
    int32_t Q [ ],
    double Dx [ ], double Dz [ ],
    int32_t *do_recip,
    double Rs [ ],
    void *Numeric
) ;

int umfpack_zl_get_numeric
(
    int64_t Lp [ ],
    int64_t Lj [ ],
    double Lx [ ], double Lz [ ],
    int64_t Up [ ],
    int64_t Ui [ ],
    double Ux [ ], double Uz [ ],
    int64_t P [ ],
    int64_t Q [ ],
    double Dx [ ], double Dz [ ],
    int64_t *do_recip,
    double Rs [ ],
    void *Numeric
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t *Lp, *Lj, *Up, *Ui, *P, *Q, do_recip ;
    double *Lx, *Ux, *Dx, *Rs ;
    int status = umfpack_di_get_numeric (Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx,
        &do_recip, Rs, Numeric) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t *Lp, *Lj, *Up, *Ui, *P, *Q, do_recip ;
    double *Lx, *Ux, *Dx, *Rs ;
    int status = umfpack_dl_get_numeric (Lp, Lj, Lx, Up, Ui, Ux, P, Q, Dx,
        &do_recip, Rs, Numeric) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int32_t *Lp, *Lj, *Up, *Ui, *P, *Q, do_recip ;
    double *Lx, *Lz, *Ux, *Uz, *Dx, *Dz, *Rs ;
    int status = umfpack_zi_get_numeric (Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q,
        Dx, Dz, &do_recip, Rs, Numeric) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    int64_t *Lp, *Lj, *Up, *Ui, *P, *Q, do_recip ;
    double *Lx, *Lz, *Ux, *Uz, *Dx, *Dz, *Rs ;
    int status = umfpack_zl_get_numeric (Lp, Lj, Lx, Lz, Up, Ui, Ux, Uz, P, Q,
        Dx, Dz, &do_recip, Rs, Numeric) ;

packed complex int32_t/int64_t Syntax:

    Same as above, except Lz, Uz, and Dz are all NULL.

Purpose:

    This routine copies the LU factors and permutation vectors from the Numeric
    object into user-accessible arrays.  This routine is not needed to solve a
    linear system.  Note that the output arrays Lp, Lj, Lx, Up, Ui, Ux, P, Q,
    Dx, and Rs are not allocated by umfpack_*_get_numeric; they must exist on
    input.

    All output arguments are optional.  If any of them are NULL
    on input, then that part of the LU factorization is not copied.  You can
    use this routine to extract just the parts of the LU factorization that
    you want.  For example, to retrieve just the column permutation Q, use:

    status = umfpack_di_get_numeric (NULL, NULL, NULL, NULL, NULL, NULL, NULL,
        Q, NULL, NULL, NULL, Numeric) ;

Returns:

    Returns UMFPACK_OK if successful.  Returns UMFPACK_ERROR_out_of_memory
    if insufficient memory is available for the 2*max(n_row,n_col) integer
    workspace that umfpack_*_get_numeric allocates to construct L and/or U.
    Returns UMFPACK_ERROR_invalid_Numeric_object if the Numeric object provided
    as input is invalid.

Arguments:

    Int Lp [n_row+1] ;  Output argument.
    Int Lj [lnz] ;      Output argument.
    double Lx [lnz] ;   Output argument.  Size 2*lnz for packed complex case.
    double Lz [lnz] ;   Output argument for complex versions.

        The n_row-by-min(n_row,n_col) matrix L is returned in compressed-row
        form.  The column indices of row i and corresponding numerical values
        are in:

            Lj [Lp [i] ... Lp [i+1]-1]
            Lx [Lp [i] ... Lp [i+1]-1]  real part
            Lz [Lp [i] ... Lp [i+1]-1]  imaginary part (complex versions)

        respectively.  Each row is stored in sorted order, from low column
        indices to higher.  The last entry in each row is the diagonal, which
        is numerically equal to one.  The sizes of Lp, Lj, Lx, and Lz are
        returned by umfpack_*_get_lunz.    If Lp, Lj, or Lx are not present,
        then the matrix L is not returned.  This is not an error condition.
        The L matrix can be printed if n_row, Lp, Lj, Lx (and Lz for the split
        complex case) are passed to umfpack_*_report_matrix (using the
        "row" form).

        If Lx is present and Lz is NULL, then both real
        and imaginary parts are returned in Lx[0..2*lnz-1], with Lx[2*k]
        and Lx[2*k+1] being the real and imaginary part of the kth entry.

    Int Up [n_col+1] ;  Output argument.
    Int Ui [unz] ;      Output argument.
    double Ux [unz] ;   Output argument. Size 2*unz for packed complex case.
    double Uz [unz] ;   Output argument for complex versions.

        The min(n_row,n_col)-by-n_col matrix U is returned in compressed-column
        form.  The row indices of column j and corresponding numerical values
        are in

            Ui [Up [j] ... Up [j+1]-1]
            Ux [Up [j] ... Up [j+1]-1]  real part
            Uz [Up [j] ... Up [j+1]-1]  imaginary part (complex versions)

        respectively.  Each column is stored in sorted order, from low row
        indices to higher.  The last entry in each column is the diagonal
        (assuming that it is nonzero).  The sizes of Up, Ui, Ux, and Uz are
        returned by umfpack_*_get_lunz.  If Up, Ui, or Ux are not present,
        then the matrix U is not returned.  This is not an error condition.
        The U matrix can be printed if n_col, Up, Ui, Ux (and Uz for the
        split complex case) are passed to umfpack_*_report_matrix (using the
        "column" form).

        If Ux is present and Uz is NULL, then both real
        and imaginary parts are returned in Ux[0..2*unz-1], with Ux[2*k]
        and Ux[2*k+1] being the real and imaginary part of the kth entry.

    Int P [n_row] ;             Output argument.

        The permutation vector P is defined as P [k] = i, where the original
        row i of A is the kth pivot row in PAQ.  If you do not want the P vector
        to be returned, simply pass (Int *) NULL for P.  This is not an error
        condition.  You can print P and Q with umfpack_*_report_perm.

    Int Q [n_col] ;             Output argument.

        The permutation vector Q is defined as Q [k] = j, where the original
        column j of A is the kth pivot column in PAQ.  If you not want the Q
        vector to be returned, simply pass (Int *) NULL for Q.  This is not
        an error condition.  Note that Q is not necessarily identical to
        Qtree, the column pre-ordering held in the Symbolic object.  Refer to
        the description of Qtree and Front_npivcol in umfpack_*_get_symbolic for
        details.

    double Dx [min(n_row,n_col)] ;      Output argument.  Size 2*n for
                                        the packed complex case.
    double Dz [min(n_row,n_col)] ;      Output argument for complex versions.

        The diagonal of U is also returned in Dx and Dz.  You can extract the
        diagonal of U without getting all of U by passing a non-NULL Dx (and
        Dz for the complex version) and passing Up, Ui, and Ux as NULL.  Dx is
        the real part of the diagonal, and Dz is the imaginary part.

        If Dx is present and Dz is NULL, then both real
        and imaginary parts are returned in Dx[0..2*min(n_row,n_col)-1],
        with Dx[2*k] and Dx[2*k+1] being the real and imaginary part of the kth
        entry.

    Int *do_recip ;             Output argument.

        This argument defines how the scale factors Rs are to be interpretted.

        If do_recip is TRUE (one), then the scale factors Rs [i] are to be used
        by multiplying row i by Rs [i].  Otherwise, the entries in row i are to
        be divided by Rs [i].

        If UMFPACK has been compiled with gcc, or for MATLAB as either a
        built-in routine or as a mexFunction, then the NRECIPROCAL flag is
        set, and do_recip will always be FALSE (zero).

    double Rs [n_row] ;         Output argument.

        The row scale factors are returned in Rs [0..n_row-1].  Row i of A is
        scaled by dividing or multiplying its values by Rs [i].  If default
        scaling is in use, Rs [i] is the sum of the absolute values of row i
        (or its reciprocal).  If max row scaling is in use, then Rs [i] is the
        maximum absolute value in row i (or its reciprocal).
        Otherwise, Rs [i] = 1.  If row i is all zero, Rs [i] = 1 as well.  For
        the complex version, an approximate absolute value is used
        (|x_real|+|x_imag|).

    void *Numeric ;     Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric.
*/

//------------------------------------------------------------------------------
// umfpack_get_symbolic
//------------------------------------------------------------------------------

int umfpack_di_get_symbolic
(
    int32_t *n_row,
    int32_t *n_col,
    int32_t *n1,
    int32_t *nz,
    int32_t *nfr,
    int32_t *nchains,
    int32_t P [ ],
    int32_t Q [ ],
    int32_t Front_npivcol [ ],
    int32_t Front_parent [ ],
    int32_t Front_1strow [ ],
    int32_t Front_leftmostdesc [ ],
    int32_t Chain_start [ ],
    int32_t Chain_maxrows [ ],
    int32_t Chain_maxcols [ ],
    int32_t Dmap [ ],               // added for v6.0.0
    void *Symbolic
) ;

int umfpack_dl_get_symbolic
(
    int64_t *n_row,
    int64_t *n_col,
    int64_t *n1,
    int64_t *nz,
    int64_t *nfr,
    int64_t *nchains,
    int64_t P [ ],
    int64_t Q [ ],
    int64_t Front_npivcol [ ],
    int64_t Front_parent [ ],
    int64_t Front_1strow [ ],
    int64_t Front_leftmostdesc [ ],
    int64_t Chain_start [ ],
    int64_t Chain_maxrows [ ],
    int64_t Chain_maxcols [ ],
    int64_t Dmap [ ],  // added for v6.0.0
    void *Symbolic
) ;

int umfpack_zi_get_symbolic
(
    int32_t *n_row,
    int32_t *n_col,
    int32_t *n1,
    int32_t *nz,
    int32_t *nfr,
    int32_t *nchains,
    int32_t P [ ],
    int32_t Q [ ],
    int32_t Front_npivcol [ ],
    int32_t Front_parent [ ],
    int32_t Front_1strow [ ],
    int32_t Front_leftmostdesc [ ],
    int32_t Chain_start [ ],
    int32_t Chain_maxrows [ ],
    int32_t Chain_maxcols [ ],
    int32_t Dmap [ ],               // added for v6.0.0
    void *Symbolic
) ;

int umfpack_zl_get_symbolic
(
    int64_t *n_row,
    int64_t *n_col,
    int64_t *n1,
    int64_t *nz,
    int64_t *nfr,
    int64_t *nchains,
    int64_t P [ ],
    int64_t Q [ ],
    int64_t Front_npivcol [ ],
    int64_t Front_parent [ ],
    int64_t Front_1strow [ ],
    int64_t Front_leftmostdesc [ ],
    int64_t Chain_start [ ],
    int64_t Chain_maxrows [ ],
    int64_t Chain_maxcols [ ],
    int64_t Dmap [ ],  // added for v6.0.0
    void *Symbolic
) ;

/*

double int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, nz, nfr, nchains, *P, *Q,
        *Front_npivcol, *Front_parent, *Front_1strow, *Front_leftmostdesc,
        *Chain_start, *Chain_maxrows, *Chain_maxcols, *Dmap ;
    void *Symbolic ;
    int status = umfpack_di_get_symbolic (&n_row, &n_col, &nz, &nfr, &nchains,
        P, Q, Front_npivcol, Front_parent, Front_1strow,
        Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols,
        Dmap, Symbolic) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, nz, nfr, nchains, *P, *Q,
        *Front_npivcol, *Front_parent, *Front_1strow, *Front_leftmostdesc,
        *Chain_start, *Chain_maxrows, *Chain_maxcols, *Dmap ;
    void *Symbolic ;
    int status = umfpack_dl_get_symbolic (&n_row, &n_col, &nz, &nfr, &nchains,
        P, Q, Front_npivcol, Front_parent, Front_1strow,
        Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols,
        Dmap, Symbolic) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, nz, nfr, nchains, *P, *Q,
        *Front_npivcol, *Front_parent, *Front_1strow, *Front_leftmostdesc,
        *Chain_start, *Chain_maxrows, *Chain_maxcols, *Dmap ;
    void *Symbolic ;
    int status = umfpack_zi_get_symbolic (&n_row, &n_col, &nz, &nfr, &nchains,
        P, Q, Front_npivcol, Front_parent, Front_1strow,
        Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols,
        Dmap, Symbolic) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, nz, nfr, nchains, *P, *Q,
        *Front_npivcol, *Front_parent, *Front_1strow, *Front_leftmostdesc,
        *Chain_start, *Chain_maxrows, *Chain_maxcols, *Dmap ;
    void *Symbolic ;
    int status = umfpack_zl_get_symbolic (&n_row, &n_col, &nz, &nfr, &nchains,
        P, Q, Front_npivcol, Front_parent, Front_1strow,
        Front_leftmostdesc, Chain_start, Chain_maxrows, Chain_maxcols,
        Dmap, Symbolic) ;

Purpose:

    Copies the contents of the Symbolic object into simple integer arrays
    accessible to the user.  This routine is not needed to factorize and/or
    solve a sparse linear system using UMFPACK.  Note that the output arrays
    P, Q, Front_npivcol, Front_parent, Front_1strow, Front_leftmostdesc,
    Chain_start, Chain_maxrows, and Chain_maxcols are not allocated by
    umfpack_*_get_symbolic; they must exist on input.

    All output arguments are optional.  If any of them are NULL
    on input, then that part of the symbolic analysis is not copied.  You can
    use this routine to extract just the parts of the symbolic analysis that
    you want.  For example, to retrieve just the column permutation Q, use:

    status = umfpack_di_get_symbolic (NULL, NULL, NULL, NULL, NULL, NULL, NULL,
            Q, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, Symbolic) ;

    The only required argument the last one, the pointer to the Symbolic object.

    The Symbolic object is small.  Its size for an n-by-n square matrix varies
    from 4*n to 13*n, depending on the matrix.  The object holds the initial
    column permutation, the supernodal column elimination tree, and information
    about each frontal matrix.  You can print it with umfpack_*_report_symbolic.

Returns:

    Returns UMFPACK_OK if successful, UMFPACK_ERROR_invalid_Symbolic_object
    if Symbolic is an invalid object.

Arguments:

    Int *n_row ;        Output argument.
    Int *n_col ;        Output argument.

        The dimensions of the matrix A analyzed by the call to
        umfpack_*_symbolic that generated the Symbolic object.

    Int *n1 ;           Output argument.

        The number of pivots with zero Markowitz cost (they have just one entry
        in the pivot row, or the pivot column, or both).  These appear first in
        the output permutations P and Q.

    Int *nz ;           Output argument.

        The number of nonzeros in A.

    Int *nfr ;  Output argument.

        The number of frontal matrices that will be used by umfpack_*_numeric
        to factorize the matrix A.  It is in the range 0 to n_col.

    Int *nchains ;      Output argument.

        The frontal matrices are related to one another by the supernodal
        column elimination tree.  Each node in this tree is one frontal matrix.
        The tree is partitioned into a set of disjoint paths, and a frontal
        matrix chain is one path in this tree.  Each chain is factorized using
        a unifrontal technique, with a single working array that holds each
        frontal matrix in the chain, one at a time.  nchains is in the range
        0 to nfr.

    Int P [n_row] ;     Output argument.

        The initial row permutation.  If P [k] = i, then this means that
        row i is the kth row in the pre-ordered matrix.  In general, this P is
        not the same as the final row permutation computed by umfpack_*_numeric.

        For the unsymmetric strategy, P defines the row-merge order.  Let j be
        the column index of the leftmost nonzero entry in row i of A*Q.  Then
        P defines a sort of the rows according to this vastatus, lue.  A row can appear
        earlier in this ordering if it is aggressively absorbed before it can
        become a pivot row.  If P [k] = i, row i typically will not be the kth
        pivot row.

        For the symmetric strategy, P = Q.  If no pivoting occurs during
        numerical factorization, P [k] = i also defines the final permutation
        of umfpack_*_numeric, for the symmetric strategy.

    Int Q [n_col] ;     Output argument.

        The initial column permutation.  If Q [k] = j, then this means that
        column j is the kth pivot column in the pre-ordered matrix.  Q is
        not necessarily the same as the final column permutation Q, computed by
        umfpack_*_numeric.  The numeric factorization may reorder the pivot
        columns within each frontal matrix to reduce fill-in.  If the matrix is
        structurally singular, and if the symmetric strategy is
        used (or if Control [UMFPACK_FIXQ] > 0), then this Q will be the same
        as the final column permutation computed in umfpack_*_numeric.

    Int Front_npivcol [n_col+1] ;       Output argument.

        This array should be of size at least n_col+1, in order to guarantee
        that it will be large enough to hold the output.  Only the first nfr+1
        entries are used, however.

        The kth frontal matrix holds Front_npivcol [k] pivot columns.  Thus, the
        first frontal matrix, front 0, is used to factorize the first
        Front_npivcol [0] columns; these correspond to the original columns
        Q [0] through Q [Front_npivcol [0]-1].  The next frontal matrix
        is used to factorize the next Front_npivcol [1] columns, which are thus
        the original columns Q [Front_npivcol [0]] through
        Q [Front_npivcol [0] + Front_npivcol [1] - 1], and so on.  Columns
        with no entries at all are put in a placeholder "front",
        Front_npivcol [nfr].  The sum of Front_npivcol [0..nfr] is equal to
        n_col.

        Any modifications that umfpack_*_numeric makes to the initial column
        permutation are constrained to within each frontal matrix.  Thus, for
        the first frontal matrix, Q [0] through Q [Front_npivcol [0]-1] is some
        permutation of the columns Q [0] through
        Q [Front_npivcol [0]-1].  For second frontal matrix,
        Q [Front_npivcol [0]] through Q [Front_npivcol [0] + Front_npivcol[1]-1]
        is some permutation of the same portion of Q, and so on.  All pivot
        columns are numerically factorized within the frontal matrix originally
        determined by the symbolic factorization; there is no delayed pivoting
        across frontal matrices.

    Int Front_parent [n_col+1] ;        Output argument.

        This array should be of size at least n_col+1, in order to guarantee
        that it will be large enough to hold the output.  Only the first nfr+1
        entries are used, however.

        Front_parent [0..nfr] holds the supernodal column elimination tree
        (including the placeholder front nfr, which may be empty).  Each node in
        the tree corresponds to a single frontal matrix.  The parent of node f
        is Front_parent [f].

    Int Front_1strow [n_col+1] ;        Output argument.

        This array should be of size at least n_col+1, in order to guarantee
        that it will be large enough to hold the output.  Only the first nfr+1
        entries are used, however.

        Front_1strow [k] is the row index of the first row in A (P,Q)
        whose leftmost entry is in a pivot column for the kth front.  This is
        necessary only to properly factorize singular matrices.  Rows in the
        range Front_1strow [k] to Front_1strow [k+1]-1 first become pivot row
        candidates at the kth front.  Any rows not eliminated in the kth front
        may be selected as pivot rows in the parent of k (Front_parent [k])
        and so on up the tree.

    Int Front_leftmostdesc [n_col+1] ;  Output argument.

        This array should be of size at least n_col+1, in order to guarantee
        that it will be large enough to hold the output.  Only the first nfr+1
        entries are used, however.

        Front_leftmostdesc [k] is the leftmost descendant of front k, or k
        if the front has no children in the tree.  Since the rows and columns
        (P and Q) have been post-ordered via a depth-first-search of
        the tree, rows in the range Front_1strow [Front_leftmostdesc [k]] to
        Front_1strow [k+1]-1 form the entire set of candidate pivot rows for
        the kth front (some of these will typically have already been selected
        by fronts in the range Front_leftmostdesc [k] to front k-1, before
        the factorization reaches front k).

    Chain_start [n_col+1] ;     Output argument.

        This array should be of size at least n_col+1, in order to guarantee
        that it will be large enough to hold the output.  Only the first
        nchains+1 entries are used, however.

        The kth frontal matrix chain consists of frontal matrices Chain_start[k]
        through Chain_start [k+1]-1.  Thus, Chain_start [0] is always 0, and
        Chain_start [nchains] is the total number of frontal matrices, nfr.  For
        two adjacent fronts f and f+1 within a single chain, f+1 is always the
        parent of f (that is, Front_parent [f] = f+1).

    Int Chain_maxrows [n_col+1] ;       Output argument.
    Int Chain_maxcols [n_col+1] ;       Output argument.

        These arrays should be of size at least n_col+1, in order to guarantee
        that they will be large enough to hold the output.  Only the first
        nchains entries are used, however.

        The kth frontal matrix chain requires a single working array of
        dimension Chain_maxrows [k] by Chain_maxcols [k], for the unifrontal
        technique that factorizes the frontal matrix chain.  Since the symbolic
        factorization only provides an upper bound on the size of each frontal
        matrix, not all of the working array is necessarily used during the
        numerical factorization.

        Note that the upper bound on the number of rows and columns of each
        frontal matrix is computed by umfpack_*_symbolic, but all that is
        required by umfpack_*_numeric is the maximum of these two sets of
        values for each frontal matrix chain.  Thus, the size of each
        individual frontal matrix is not preserved in the Symbolic object.

    Int Dmap [n_col+1] ;                Output argument (new to v6.0.0)

        For the symmetric strategy, the initial permutation may in some cases
        be unsymmetric.  This is handled by creating a diagonal map array,
        Dmap.  Suppose S = P*A*Q is the matrix permuted before the matrix is
        factorized.  Then Dmap [j] = i if entry S(i,j) corresponds to an
        original diagonal entry of A.  If there is no diagonal map, then Dmap
        [i] = i for all i.  In v5.x, the Dmap was incorporated into P.  In
        v6.0.0, P is no modified by Dmap, and Dmap is returned as its own
        array.

    void *Symbolic ;                    Input argument, not modified.

        The Symbolic object, which holds the symbolic factorization computed by
        umfpack_*_symbolic.  The Symbolic object is not modified by
        umfpack_*_get_symbolic.
*/

//------------------------------------------------------------------------------
// umfpack_save_numeric
//------------------------------------------------------------------------------

int umfpack_di_save_numeric
(
    void *Numeric,
    char *filename
) ;

int umfpack_dl_save_numeric
(
    void *Numeric,
    char *filename
) ;

int umfpack_zi_save_numeric
(
    void *Numeric,
    char *filename
) ;

int umfpack_zl_save_numeric
(
    void *Numeric,
    char *filename
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_di_save_numeric (Numeric, filename) ;

double int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_dl_save_numeric (Numeric, filename) ;

complex int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_zi_save_numeric (Numeric, filename) ;

complex int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_zl_save_numeric (Numeric, filename) ;

Purpose:

    Saves a Numeric object to a file, which can later be read by
    umfpack_*_load_numeric.  The Numeric object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Numeric_object if Numeric is not valid.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void *Numeric ;         Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric or loaded by umfpack_*_load_numeric.

    char *filename ;        Input argument, not modified.

        A string that contains the filename to which the Numeric
        object is written.
*/

//------------------------------------------------------------------------------
// umfpack_load_numeric
//------------------------------------------------------------------------------

int umfpack_di_load_numeric
(
    void **Numeric,
    char *filename
) ;

int umfpack_dl_load_numeric
(
    void **Numeric,
    char *filename
) ;

int umfpack_zi_load_numeric
(
    void **Numeric,
    char *filename
) ;

int umfpack_zl_load_numeric
(
    void **Numeric,
    char *filename
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_di_load_numeric (&Numeric, filename) ;

double int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_dl_load_numeric (&Numeric, filename) ;

complex int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_zi_load_numeric (&Numeric, filename) ;

complex int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Numeric ;
    int status = umfpack_zl_load_numeric (&Numeric, filename) ;

Purpose:

    Loads a Numeric object from a file created by umfpack_*_save_numeric.  The
    Numeric handle passed to this routine is overwritten with the new object.
    If that object exists prior to calling this routine, a memory leak will
    occur.  The contents of Numeric are ignored on input.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void **Numeric ;        Output argument.

        **Numeric is the address of a (void *) pointer variable in the user's
        calling routine (see Syntax, above).  On input, the contents of this
        variable are not defined.  On output, this variable holds a (void *)
        pointer to the Numeric object (if successful), or (void *) NULL if
        a failure occurred.

    char *filename ;        Input argument, not modified.

        A string that contains the filename from which to read the Numeric
        object.
*/

//------------------------------------------------------------------------------
// umfpack_copy_numeric
//------------------------------------------------------------------------------

int umfpack_di_copy_numeric
(
    void **Numeric,
    void *Original
) ;

int umfpack_dl_copy_numeric
(
    void **Numeric,
    void *Original
) ;

int umfpack_zi_copy_numeric
(
    void **Numeric,
    void *Original
) ;

int umfpack_zl_copy_numeric
(
    void **Numeric,
    void *Original
) ;

/*
double int Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Numeric ;
    status = umfpack_di_copy_numeric (&Numeric, Original) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Numeric ;
    int status = umfpack_dl_copy_numeric (&Numeric, Original) ;

complex int Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Numeric ;
    int status = umfpack_zi_copy_numeric (&Numeric, Original) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Numeric ;
    int status = umfpack_zl_copy_numeric (&Numeric, Original) ;

Purpose:

    Copies a Numeric object.  The Numeric handle passed to this routine is
    overwritten with the new object.  If that object exists prior to calling
    this routine, a memory leak will occur.  The contents of numeric are
    ignored on input.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.

Arguments:

    void **Numeric ;        Output argument.

        **Numeric is the address of a (void *) pointer variable in the user's
        calling routine (see Syntax, above).  On input, the contents of this
        variable are not defined.  On output, this variable holds a (void *)
        pointer to the numeric object (if successful), or (void *) NULL if
        a failure occurred.

    void *Original ;        Input argument, not modified.

        The Numeric object to be copied.
*/

//------------------------------------------------------------------------------
// umfpack_serialize_numeric_size
//------------------------------------------------------------------------------

int umfpack_di_serialize_numeric_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Numeric               // input: Numeric object to serialize
) ;

int umfpack_dl_serialize_numeric_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Numeric               // input: Numeric object to serialize
) ;

int umfpack_zi_serialize_numeric_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Numeric               // input: Numeric object to serialize
) ;

int umfpack_zl_serialize_numeric_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Numeric               // input: Numeric object to serialize
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Numeric ;
    int status = umfpack_di_serialize_numeric_size (&blobsize, Numeric) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Numeric ;
    int status = umfpack_dl_serialize_numeric_size (&blobsize, Numeric) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Numeric ;
    int status = umfpack_zi_serialize_numeric_size (&blobsize, Numeric) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Numeric ;
    int status = umfpack_zl_serialize_numeric_size (&blobsize, Numeric) ;

Purpose:

    Determines the required size of the serialized "blob" for the input to
    umfpack_*_serialize_numeric.  The Numeric object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Numeric_object if Numeric is not valid.
    UMFPACK_ERROR_argument_missing if blobsize is NULL.

Arguments:

    int64_t *blobsize ;     Output argument.

        Required size of the blob to hold the Numeric object, in bytes.

    void *Numeric ;         Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric or created by umfpack_*_deserialize_numeric,
        umfpack_*_load_numeric, or umfpack_*_copy_numeric.
*/

//------------------------------------------------------------------------------
// umfpack_serialize_numeric
//------------------------------------------------------------------------------

int umfpack_di_serialize_numeric
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Numeric           // input: Numeric object to serialize
) ;

int umfpack_dl_serialize_numeric
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Numeric           // input: Numeric object to serialize
) ;

int umfpack_zi_serialize_numeric
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Numeric           // input: Numeric object to serialize
) ;

int umfpack_zl_serialize_numeric
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Numeric           // input: Numeric object to serialize
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_di_serialize_numeric (blob, blobsize, Numeric) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_dl_serialize_numeric (blob, blobsize, Numeric) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_zi_serialize_numeric (blob, blobsize, Numeric) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_zl_serialize_numeric (blob, blobsize, Numeric) ;

Purpose:

    Copies the contents of a Numeric object into the serialized "blob".
    Numeric object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_argument_missing if blob or Numeric are NULL.
    UMFPACK_ERROR_invalid_Numeric_object if Numeric is not valid.
    UMFPACK_ERROR_invalid_blob if blob is too small.

Arguments:

    int8_t *blob ;          Output  argument.

        A user-allocated array of size blobsize.  On output, it contains the
        serialized blob created from the Numeric object.

    int64_t blobsize ;      Input argument, not modified.

        Size of the blob, in bytes.  Must be at least as large as the
        value returned by umfpack_*_serialize_numeric_size.

    void *Numeric ;         Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric or created by umfpack_*_deserialize_numeric,
        umfpack_*_load_numeric, or umfpack_*_copy_numeric.
*/

//------------------------------------------------------------------------------
// umfpack_deserialize_numeric
//------------------------------------------------------------------------------

int umfpack_di_deserialize_numeric
(
    void **Numeric,         // output: Numeric object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

int umfpack_dl_deserialize_numeric
(
    void **Numeric,         // output: Numeric object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

int umfpack_zi_deserialize_numeric
(
    void **Numeric,         // output: Numeric object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

int umfpack_zl_deserialize_numeric
(
    void **Numeric,         // output: Numeric object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_di_deserialize_numeric (&Numeric, blob, blobsize) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_dl_deserialize_numeric (&Numeric, blob, blobsize) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_zi_deserialize_numeric (&Numeric, blob, blobsize) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Numeric ;
    int status = umfpack_zl_deserialize_numeric (&Numeric, blob, blobsize) ;

Purpose:

    Constructs a new Numeric object from the serialized "blob".
    The blob is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_argument_missing if blob or Numeric are NULL.
    UMFPACK_ERROR_invalid_Numeric_object if Numeric is not valid.
    UMFPACK_ERROR_invalid_blob if blob is too small.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.

Arguments:

    void **Numeric ;        Input argument, not modified.

        On input, the contents of this variable are not defined.  On output,
        this variable holds a (void *) pointer to the Numeric object (if
        successful), or (void *) NULL if a failure occurred.

    int8_t *blob ;          Input argument, not modified.

        A user-allocated array of size blobsize containing a blob created
        by umfpack_*_serialize_numeric.

    int64_t blobsize ;      Input argument, not modified.

        Size of the blob, in bytes.
*/

//------------------------------------------------------------------------------
// umfpack_save_symbolic
//------------------------------------------------------------------------------

int umfpack_di_save_symbolic
(
    void *Symbolic,
    char *filename
) ;

int umfpack_dl_save_symbolic
(
    void *Symbolic,
    char *filename
) ;

int umfpack_zi_save_symbolic
(
    void *Symbolic,
    char *filename
) ;

int umfpack_zl_save_symbolic
(
    void *Symbolic,
    char *filename
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_di_save_symbolic (Symbolic, filename) ;

double int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_dl_save_symbolic (Symbolic, filename) ;

complex int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_zi_save_symbolic (Symbolic, filename) ;

complex int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_zl_save_symbolic (Symbolic, filename) ;

Purpose:

    Saves a Symbolic object to a file, which can later be read by
    umfpack_*_load_symbolic.  The Symbolic object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Symbolic_object if Symbolic is not valid.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void *Symbolic ;        Input argument, not modified.

        Symbolic must point to a valid Symbolic object, computed by
        umfpack_*_symbolic or loaded by umfpack_*_load_symbolic.

    char *filename ;        Input argument, not modified.

        A string that contains the filename to which the Symbolic
        object is written.
*/

//------------------------------------------------------------------------------
// umfpack_load_symbolic
//------------------------------------------------------------------------------

int umfpack_di_load_symbolic
(
    void **Symbolic,
    char *filename
) ;

int umfpack_dl_load_symbolic
(
    void **Symbolic,
    char *filename
) ;

int umfpack_zi_load_symbolic
(
    void **Symbolic,
    char *filename
) ;

int umfpack_zl_load_symbolic
(
    void **Symbolic,
    char *filename
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_di_load_symbolic (&Symbolic, filename) ;

double int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_dl_load_symbolic (&Symbolic, filename) ;

complex int32_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_zi_load_symbolic (&Symbolic, filename) ;

complex int64_t Syntax:

    #include "umfpack.h"
    char *filename ;
    void *Symbolic ;
    int status = umfpack_zl_load_symbolic (&Symbolic, filename) ;

Purpose:

    Loads a Symbolic object from a file created by umfpack_*_save_symbolic. The
    Symbolic handle passed to this routine is overwritten with the new object.
    If that object exists prior to calling this routine, a memory leak will
    occur.  The contents of Symbolic are ignored on input.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.
    UMFPACK_ERROR_file_IO if an I/O error occurred.

Arguments:

    void **Symbolic ;       Output argument.

        **Symbolic is the address of a (void *) pointer variable in the user's
        calling routine (see Syntax, above).  On input, the contents of this
        variable are not defined.  On output, this variable holds a (void *)
        pointer to the Symbolic object (if successful), or (void *) NULL if
        a failure occurred.

    char *filename ;        Input argument, not modified.

        A string that contains the filename from which to read the Symbolic
        object.
*/

//------------------------------------------------------------------------------
// umfpack_copy_symbolic
//------------------------------------------------------------------------------

int umfpack_di_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

int umfpack_dl_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

int umfpack_zi_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

int umfpack_zl_copy_symbolic
(
    void **Symbolic,
    void *Original
) ;

/*
double int Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Symbolic ;
    int status = umfpack_di_copy_symbolic (&Symbolic, Original) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Symbolic ;
    int status = umfpack_dl_copy_symbolic (&Symbolic, Original) ;

complex int Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Symbolic ;
    int status = umfpack_zi_copy_symbolic (&Symbolic, Original) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Original ;
    void *Symbolic ;
    int status = umfpack_zl_copy_symbolic (&Symbolic, Original) ;

Purpose:

    Copies a Symbolic object.  The Symbolic handle passed to this routine is
    overwritten with the new object.  If that object exists prior to calling
    this routine, a memory leak will occur.  The contents of Symbolic are
    ignored on input.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.

Arguments:

    void **Symbolic ;       Output argument.

        **Symbolic is the address of a (void *) pointer variable in the user's
        calling routine (see Syntax, above).  On input, the contents of this
        variable are not defined.  On output, this variable holds a (void *)
        pointer to the Symbolic object (if successful), or (void *) NULL if
        a failure occurred.

    void *Original ;        Input argument, not modified.

        The original Symbolic object to be copied.
*/

//------------------------------------------------------------------------------
// umfpack_serialize_symbolic_size
//------------------------------------------------------------------------------

int umfpack_di_serialize_symbolic_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Symbolic              // input: Symbolic object to serialize
) ;

int umfpack_dl_serialize_symbolic_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Symbolic              // input: Symbolic object to serialize
) ;

int umfpack_zi_serialize_symbolic_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Symbolic              // input: Symbolic object to serialize
) ;

int umfpack_zl_serialize_symbolic_size
(
    int64_t *blobsize,          // output: required size of blob
    void *Symbolic              // input: Symbolic object to serialize
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Symbolic ;
    int status = umfpack_di_serialize_symbolic_size (&blobsize, Symbolic) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Symbolic ;
    int status = umfpack_dl_serialize_symbolic_size (&blobsize, Symbolic) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Symbolic ;
    int status = umfpack_zi_serialize_symbolic_size (&blobsize, Symbolic) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    void *Symbolic ;
    int status = umfpack_zl_serialize_symbolic_size (&blobsize, Symbolic) ;

Purpose:

    Determines the required size of the serialized "blob" for the input to
    umfpack_*_serialize_symbolic.  The Symbolic object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_invalid_Symbolic_object if Symbolic is not valid.
    UMFPACK_ERROR_argument_missing if blobsize is NULL.

Arguments:

    int64_t *blobsize ;     Output argument.

        Required size of the blob to hold the Symbolic object, in bytes.

    void *Symbolic ;        Input argument, not modified.

        Symbolic must point to a valid Symbolic object, computed by
        umfpack_*_symbolic or created by umfpack_*_deserialize_symbolic,
        umfpack_*_load_symbolic, or umfpack_*_copy_symbolic.
*/

//------------------------------------------------------------------------------
// umfpack_serialize_symbolic
//------------------------------------------------------------------------------

int umfpack_di_serialize_symbolic
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Symbolic          // input: Symbolic object to serialize
) ;

int umfpack_dl_serialize_symbolic
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Symbolic          // input: Symbolic object to serialize
) ;

int umfpack_zi_serialize_symbolic
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Symbolic          // input: Symbolic object to serialize
) ;

int umfpack_zl_serialize_symbolic
(
    int8_t *blob,           // output: serialized blob of size blobsize,
                            // allocated but unitialized on input.
    int64_t blobsize,       // input: size of the blob
    void *Symbolic          // input: Symbolic object to serialize
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_di_serialize_symbolic (blob, blobsize, Symbolic) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_dl_serialize_symbolic (blob, blobsize, Symbolic) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_zi_serialize_symbolic (blob, blobsize, Symbolic) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_zl_serialize_symbolic (blob, blobsize, Symbolic) ;

Purpose:

    Copies the contents of a Symbolic object into the serialized "blob".
    Symbolic object is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_argument_missing if blob or Symbolic are NULL.
    UMFPACK_ERROR_invalid_Symbolic_object if Symbolic is not valid.
    UMFPACK_ERROR_invalid_blob if blob is too small.

Arguments:

    int8_t *blob ;          Output  argument.

        A user-allocated array of size blobsize.  On output, it contains the
        serialized blob created from the Symbolic object.

    int64_t blobsize ;      Input argument, not modified.

        Size of the blob, in bytes.  Must be at least as large as the
        value returned by umfpack_*_serialize_symbolic_size.

    void *Symbolic ;        Input argument, not modified.

        Symbolic must point to a valid Symbolic object, computed by
        umfpack_*_symbolic or created by umfpack_*_deserialize_symbolic,
        umfpack_*_load_symbolic, or umfpack_*_copy_symbolic.
*/

//------------------------------------------------------------------------------
// umfpack_deserialize_symbolic
//------------------------------------------------------------------------------

int umfpack_di_deserialize_symbolic
(
    void **Symbolic,        // output: Symbolic object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

int umfpack_dl_deserialize_symbolic
(
    void **Symbolic,        // output: Symbolic object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

int umfpack_zi_deserialize_symbolic
(
    void **Symbolic,        // output: Symbolic object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

int umfpack_zl_deserialize_symbolic
(
    void **Symbolic,        // output: Symbolic object created from the blob
    int8_t *blob,           // input: serialized blob, not modified
    int64_t blobsize        // size of the blob in bytes
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_di_deserialize_symbolic (&Symbolic, blob, blobsize) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_dl_deserialize_symbolic (&Symbolic, blob, blobsize) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_zi_deserialize_symbolic (&Symbolic, blob, blobsize) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t blobsize ;
    int8_t *blob ;
    void *Symbolic ;
    int status = umfpack_zl_deserialize_symbolic (&Symbolic, blob, blobsize) ;

Purpose:

    Constructs a new Symbolic object from the serialized "blob".
    The blob is not modified.

Returns:

    UMFPACK_OK if successful.
    UMFPACK_ERROR_argument_missing if blob or Symbolic are NULL.
    UMFPACK_ERROR_invalid_Symbolic_object if Symbolic is not valid.
    UMFPACK_ERROR_invalid_blob if blob is too small.
    UMFPACK_ERROR_out_of_memory if not enough memory is available.

Arguments:

    void **Symbolic ;       Input argument, not modified.

        On input, the contents of this variable are not defined.  On output,
        this variable holds a (void *) pointer to the Symbolic object (if
        successful), or (void *) NULL if a failure occurred.

    int8_t *blob ;          Input argument, not modified.

        A user-allocated array of size blobsize containing a blob created
        by umfpack_*_serialize_symbolic.

    int64_t blobsize ;      Input argument, not modified.

        Size of the blob, in bytes.
*/

//------------------------------------------------------------------------------
// umfpack_get_determinant
//------------------------------------------------------------------------------

int umfpack_di_get_determinant
(
    double *Mx,
    double *Ex,
    void *Numeric,
    double User_Info [UMFPACK_INFO]
) ;

int umfpack_dl_get_determinant
(
    double *Mx,
    double *Ex,
    void *Numeric,
    double User_Info [UMFPACK_INFO]
) ;

int umfpack_zi_get_determinant
(
    double *Mx,
    double *Mz,
    double *Ex,
    void *Numeric,
    double User_Info [UMFPACK_INFO]
) ;

int umfpack_zl_get_determinant
(
    double *Mx,
    double *Mz,
    double *Ex,
    void *Numeric,
    double User_Info [UMFPACK_INFO]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Mx, Ex, Info [UMFPACK_INFO] ;
    int status = umfpack_di_get_determinant (&Mx, &Ex, Numeric, Info) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Mx, Ex, Info [UMFPACK_INFO] ;
    int status = umfpack_dl_get_determinant (&Mx, &Ex, Numeric, Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Mx, Mz, Ex, Info [UMFPACK_INFO] ;
    int status = umfpack_zi_get_determinant (&Mx, &Mz, &Ex, Numeric, Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double *Mx, *Mz, *Ex, Info [UMFPACK_INFO] ;
    int status = umfpack_zl_get_determinant (&Mx, &Mz, &Ex, Numeric, Info) ;

packed complex Syntax:

    Same as above, except Mz is NULL.

Author: Contributed by David Bateman, Motorola, Paris

Purpose:

    Using the LU factors and the permutation vectors contained in the Numeric
    object, calculate the determinant of the matrix A.

    The value of the determinant can be returned in two forms, depending on
    whether Ex is NULL or not.  If Ex is NULL then the value of the determinant
    is returned on Mx and Mz for the real and imaginary parts.  However, to
    avoid over- or underflows, the determinant can be split into a mantissa
    and exponent, and the parts returned separately, in which case Ex is not
    NULL.  The actual determinant is then given by

      double det ;
      det = Mx * pow (10.0, Ex) ;

    for the double case, or

      double det [2] ;
      det [0] = Mx * pow (10.0, Ex) ;       // real part
      det [1] = Mz * pow (10.0, Ex) ;       // imaginary part

    for the complex case.  Information on if the determinant will or has
    over or under-flowed is given by Info [UMFPACK_STATUS].

    In the "packed complex" syntax, Mx [0] holds the real part and Mx [1]
    holds the imaginary part.  Mz is not used (it is NULL).

Returns:

    Returns UMFPACK_OK if sucessful.  Returns UMFPACK_ERROR_out_of_memory if
    insufficient memory is available for the n_row integer workspace that
    umfpack_*_get_determinant allocates to construct pivots from the
    permutation vectors.  Returns UMFPACK_ERROR_invalid_Numeric_object if the
    Numeric object provided as input is invalid.  Returns
    UMFPACK_WARNING_singular_matrix if the determinant is zero.  Returns
    UMFPACK_WARNING_determinant_underflow or
    UMFPACK_WARNING_determinant_overflow if the determinant has underflowed
    overflowed (for the case when Ex is NULL), or will overflow if Ex is not
    NULL and det is computed (see above) in the user program.

Arguments:

    double *Mx ;   Output argument (array of size 1, or size 2 if Mz is NULL)
    double *Mz ;   Output argument (optional)
    double *Ex ;   Output argument (optional)

        The determinant returned in mantissa/exponent form, as discussed above.
        If Mz is NULL, then both the original and imaginary parts will be
        returned in Mx. If Ex is NULL then the determinant is returned directly
        in Mx and Mz (or Mx [0] and Mx [1] if Mz is NULL), rather than in
        mantissa/exponent form.

    void *Numeric ;     Input argument, not modified.

        Numeric must point to a valid Numeric object, computed by
        umfpack_*_numeric.

    double Info [UMFPACK_INFO] ;        Output argument.

        Contains information about the calculation of the determinant. If a
        (double *) NULL pointer is passed, then no statistics are returned in
        Info (this is not an error condition).  The following statistics are
        computed in umfpack_*_determinant:

        Info [UMFPACK_STATUS]: status code.  This is also the return value,
            whether or not Info is present.

            UMFPACK_OK

                The determinant was successfully found.

            UMFPACK_ERROR_out_of_memory

                Insufficient memory to solve the linear system.

            UMFPACK_ERROR_argument_missing

                Mx is missing (NULL).

            UMFPACK_ERROR_invalid_Numeric_object

                The Numeric object is not valid.

            UMFPACK_ERROR_invalid_system

                The matrix is rectangular.  Only square systems can be
                handled.

            UMFPACK_WARNING_singluar_matrix

                The determinant is zero or NaN.  The matrix is singular.

            UMFPACK_WARNING_determinant_underflow

                When passing from mantissa/exponent form to the determinant
                an underflow has or will occur.  If the mantissa/exponent from
                of obtaining the determinant is used, the underflow will occur
                in the user program.  If the single argument method of
                obtaining the determinant is used, the underflow has already
                occurred.

            UMFPACK_WARNING_determinant_overflow

                When passing from mantissa/exponent form to the determinant
                an overflow has or will occur.  If the mantissa/exponent from
                of obtaining the determinant is used, the overflow will occur
                in the user program.  If the single argument method of
                obtaining the determinant is used, the overflow has already
                occurred.


*/

//==============================================================================
//==== Reporting routines ======================================================
//==============================================================================

//------------------------------------------------------------------------------
// umfpack_report_status
//------------------------------------------------------------------------------

void umfpack_di_report_status
(
    const double Control [UMFPACK_CONTROL],
    int status
) ;

void umfpack_dl_report_status
(
    const double Control [UMFPACK_CONTROL],
    int status
) ;

void umfpack_zi_report_status
(
    const double Control [UMFPACK_CONTROL],
    int status
) ;

void umfpack_zl_report_status
(
    const double Control [UMFPACK_CONTROL],
    int status
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    int status ;
    umfpack_di_report_status (Control, status) ;

double int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    int status ;
    umfpack_dl_report_status (Control, status) ;

complex int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    int status ;
    umfpack_zi_report_status (Control, status) ;

complex int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    int status ;
    umfpack_zl_report_status (Control, status) ;

Purpose:

    Prints the status (return value) of other umfpack_* routines.

Arguments:

    double Control [UMFPACK_CONTROL] ;   Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            0 or less: no output, even when an error occurs
            1: error messages only
            2 or more: print status, whether or not an error occurred
            4 or more: also print the UMFPACK Copyright
            6 or more: also print the UMFPACK License
            Default: 1

    int status ;                        Input argument, not modified.

        The return value from another umfpack_* routine.
*/

//------------------------------------------------------------------------------
// umfpack_report_info
//------------------------------------------------------------------------------

void umfpack_di_report_info
(
    const double Control [UMFPACK_CONTROL],
    const double Info [UMFPACK_INFO]
) ;

void umfpack_dl_report_info
(
    const double Control [UMFPACK_CONTROL],
    const double Info [UMFPACK_INFO]
) ;

void umfpack_zi_report_info
(
    const double Control [UMFPACK_CONTROL],
    const double Info [UMFPACK_INFO]
) ;

void umfpack_zl_report_info
(
    const double Control [UMFPACK_CONTROL],
    const double Info [UMFPACK_INFO]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    umfpack_di_report_info (Control, Info) ;

double int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    umfpack_dl_report_info (Control, Info) ;

complex int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    umfpack_zi_report_info (Control, Info) ;

complex int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL], Info [UMFPACK_INFO] ;
    umfpack_zl_report_info (Control, Info) ;

Purpose:

    Reports statistics from the umfpack_*_*symbolic, umfpack_*_numeric, and
    umfpack_*_*solve routines.

Arguments:

    double Control [UMFPACK_CONTROL] ;   Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            0 or less: no output, even when an error occurs
            1: error messages only
            2 or more: error messages, and print all of Info
            Default: 1

    double Info [UMFPACK_INFO] ;                Input argument, not modified.

        Info is an output argument of several UMFPACK routines.
        The contents of Info are printed on standard output.
*/

//------------------------------------------------------------------------------
// umfpack_report_control
//------------------------------------------------------------------------------

void umfpack_di_report_control
(
    const double Control [UMFPACK_CONTROL]
) ;

void umfpack_dl_report_control
(
    const double Control [UMFPACK_CONTROL]
) ;

void umfpack_zi_report_control
(
    const double Control [UMFPACK_CONTROL]
) ;

void umfpack_zl_report_control
(
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_di_report_control (Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_dl_report_control (Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_zi_report_control (Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    double Control [UMFPACK_CONTROL] ;
    umfpack_zl_report_control (Control) ;

Purpose:

    Prints the current control settings.  Note that with the default print
    level, nothing is printed.  Does nothing if Control is (double *) NULL.

Arguments:

    double Control [UMFPACK_CONTROL] ;   Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            1 or less: no output
            2 or more: print all of Control
            Default: 1
*/

//------------------------------------------------------------------------------
// umfpack_report_matrix
//------------------------------------------------------------------------------

int umfpack_di_report_matrix
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ],
    int col_form,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_dl_report_matrix
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ],
    int col_form,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zi_report_matrix
(
    int32_t n_row,
    int32_t n_col,
    const int32_t Ap [ ],
    const int32_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    int col_form,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zl_report_matrix
(
    int64_t n_row,
    int64_t n_col,
    const int64_t Ap [ ],
    const int64_t Ai [ ],
    const double Ax [ ], const double Az [ ],
    int col_form,
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, *Ap, *Ai ;
    double *Ax, Control [UMFPACK_CONTROL] ;
    int status ;
    status = umfpack_di_report_matrix (n_row, n_col, Ap, Ai, Ax, 1, Control) ;
or:
    status = umfpack_di_report_matrix (n_row, n_col, Ap, Ai, Ax, 0, Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, *Ap, *Ai ;
    double *Ax, Control [UMFPACK_CONTROL] ;
    int status ;
    status = umfpack_dl_report_matrix (n_row, n_col, Ap, Ai, Ax, 1, Control) ;
or:
    status = umfpack_dl_report_matrix (n_row, n_col, Ap, Ai, Ax, 0, Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, *Ap, *Ai ;
    double *Ax, *Az, Control [UMFPACK_CONTROL] ;
    int status ;
    status = umfpack_zi_report_matrix (n_row, n_col, Ap, Ai, Ax, Az, 1,
        Control) ;
or:
    status = umfpack_zi_report_matrix (n_row, n_col, Ap, Ai, Ax, Az, 0,
        Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, *Ap, *Ai ;
    double *Ax, Control [UMFPACK_CONTROL] ;
    int status ;
    status = umfpack_zl_report_matrix (n_row, n_col, Ap, Ai, Ax, Az, 1,
        Control) ;
or:
    status = umfpack_zl_report_matrix (n_row, n_col, Ap, Ai, Ax, Az, 0,
        Control) ;

packed complex Syntax:

    Same as above, except Az is NULL.

Purpose:

    Verifies and prints a row or column-oriented sparse matrix.

Returns:

    UMFPACK_OK if Control [UMFPACK_PRL] <= 2 (the input is not checked).

    Otherwise (where n is n_col for the column form and n_row for row
    and let ni be n_row for the column form and n_col for row):

    UMFPACK_OK if the matrix is valid.

    UMFPACK_ERROR_n_nonpositive if n_row <= 0 or n_col <= 0.
    UMFPACK_ERROR_argument_missing if Ap and/or Ai are missing.
    UMFPACK_ERROR_invalid_matrix if Ap [n] < 0, if Ap [0] is not zero,
        if Ap [j+1] < Ap [j] for any j in the range 0 to n-1,
        if any row index in Ai is not in the range 0 to ni-1, or
        if the row indices in any column are not in
        ascending order, or contain duplicates.
    UMFPACK_ERROR_out_of_memory if out of memory.

Arguments:

    Int n_row ;         Input argument, not modified.
    Int n_col ;         Input argument, not modified.

        A is an n_row-by-n_row matrix.  Restriction: n_row > 0 and n_col > 0.

    Int Ap [n+1] ;      Input argument, not modified.

        n is n_row for a row-form matrix, and n_col for a column-form matrix.

        Ap is an integer array of size n+1.  If col_form is true (nonzero),
        then on input, it holds the "pointers" for the column form of the
        sparse matrix A.  The row indices of column j of the matrix A are held
        in Ai [(Ap [j]) ... (Ap [j+1]-1)].  Otherwise, Ap holds the
        row pointers, and the column indices of row j of the matrix are held
        in Ai [(Ap [j]) ... (Ap [j+1]-1)].

        The first entry, Ap [0], must be zero, and Ap [j] <= Ap [j+1] must hold
        for all j in the range 0 to n-1.  The value nz = Ap [n] is thus the
        total number of entries in the pattern of the matrix A.

    Int Ai [nz] ;       Input argument, not modified, of size nz = Ap [n].

        If col_form is true (nonzero), then the nonzero pattern (row indices)
        for column j is stored in Ai [(Ap [j]) ... (Ap [j+1]-1)].  Row indices
        must be in the range 0 to n_row-1 (the matrix is 0-based).

        Otherwise, the nonzero pattern (column indices) for row j is stored in
        Ai [(Ap [j]) ... (Ap [j+1]-1)]. Column indices must be in the range 0
        to n_col-1 (the matrix is 0-based).

    double Ax [nz] ;    Input argument, not modified, of size nz = Ap [n].
                        Size 2*nz for packed complex case.

        The numerical values of the sparse matrix A.

        If col_form is true (nonzero), then the nonzero pattern (row indices)
        for column j is stored in Ai [(Ap [j]) ... (Ap [j+1]-1)], and the
        corresponding (real) numerical values are stored in
        Ax [(Ap [j]) ... (Ap [j+1]-1)].  The imaginary parts are stored in
        Az [(Ap [j]) ... (Ap [j+1]-1)], for the complex versions
        (see below if Az is NULL).

        Otherwise, the nonzero pattern (column indices) for row j
        is stored in Ai [(Ap [j]) ... (Ap [j+1]-1)], and the corresponding
        (real) numerical values are stored in Ax [(Ap [j]) ... (Ap [j+1]-1)].
        The imaginary parts are stored in Az [(Ap [j]) ... (Ap [j+1]-1)],
        for the complex versions (see below if Az is NULL).

        No numerical values are printed if Ax is NULL.

    double Az [nz] ;    Input argument, not modified, for complex versions.

        The imaginary values of the sparse matrix A.   See the description
        of Ax, above.

        If Az is NULL, then both real
        and imaginary parts are contained in Ax[0..2*nz-1], with Ax[2*k]
        and Ax[2*k+1] being the real and imaginary part of the kth entry.

    Int col_form ;      Input argument, not modified.

        The matrix is in row-oriented form if form is col_form is false (0).
        Otherwise, the matrix is in column-oriented form.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            2 or less: no output.  returns silently without checking anything.
            3: fully check input, and print a short summary of its status
            4: as 3, but print first few entries of the input
            5: as 3, but print all of the input
            Default: 1
*/

//------------------------------------------------------------------------------
// umfpack_report_triplet
//------------------------------------------------------------------------------

int umfpack_di_report_triplet
(
    int32_t n_row,
    int32_t n_col,
    int32_t nz,
    const int32_t Ti [ ],
    const int32_t Tj [ ],
    const double Tx [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_dl_report_triplet
(
    int64_t n_row,
    int64_t n_col,
    int64_t nz,
    const int64_t Ti [ ],
    const int64_t Tj [ ],
    const double Tx [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zi_report_triplet
(
    int32_t n_row,
    int32_t n_col,
    int32_t nz,
    const int32_t Ti [ ],
    const int32_t Tj [ ],
    const double Tx [ ], const double Tz [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zl_report_triplet
(
    int64_t n_row,
    int64_t n_col,
    int64_t nz,
    const int64_t Ti [ ],
    const int64_t Tj [ ],
    const double Tx [ ], const double Tz [ ],
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, nz, *Ti, *Tj ;
    double *Tx, Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_report_triplet (n_row, n_col, nz, Ti, Tj, Tx,
        Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, nz, *Ti, *Tj ;
    double *Tx, Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_report_triplet (n_row, n_col, nz, Ti, Tj, Tx,
        Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n_row, n_col, nz, *Ti, *Tj ;
    double *Tx, *Tz, Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_report_triplet (n_row, n_col, nz, Ti, Tj, Tx, Tz,
        Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n_row, n_col, nz, *Ti, *Tj ;
    double *Tx, *Tz, Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_report_triplet (n_row, n_col, nz, Ti, Tj, Tx, Tz,
        Control) ;

packed complex Syntax:

    Same as above, except Tz is NULL.

Purpose:

    Verifies and prints a matrix in triplet form.

Returns:

    UMFPACK_OK if Control [UMFPACK_PRL] <= 2 (the input is not checked).

    Otherwise:

    UMFPACK_OK if the Triplet matrix is OK.
    UMFPACK_ERROR_argument_missing if Ti and/or Tj are missing.
    UMFPACK_ERROR_n_nonpositive if n_row <= 0 or n_col <= 0.
    UMFPACK_ERROR_invalid_matrix if nz < 0, or
        if any row or column index in Ti and/or Tj
        is not in the range 0 to n_row-1 or 0 to n_col-1, respectively.

Arguments:

    Int n_row ;         Input argument, not modified.
    Int n_col ;         Input argument, not modified.

        A is an n_row-by-n_col matrix.

    Int nz ;            Input argument, not modified.

        The number of entries in the triplet form of the matrix.

    Int Ti [nz] ;       Input argument, not modified.
    Int Tj [nz] ;       Input argument, not modified.
    double Tx [nz] ;    Input argument, not modified.
                        Size 2*nz for packed complex case.
    double Tz [nz] ;    Input argument, not modified, for complex versions.

        Ti, Tj, Tx (and Tz for complex versions) hold the "triplet" form of a
        sparse matrix.  The kth nonzero entry is in row i = Ti [k], column
        j = Tj [k], the real numerical value of a_ij is Tx [k], and the
        imaginary part of a_ij is Tz [k] (for complex versions).  The row and
        column indices i and j must be in the range 0 to n_row-1 or 0 to
        n_col-1, respectively.  Duplicate entries may be present.  The
        "triplets" may be in any order.  Tx and Tz are optional; if Tx is
        not present ((double *) NULL), then the numerical values are
        not printed.

        If Tx is present and Tz is NULL, then both real
        and imaginary parts are contained in Tx[0..2*nz-1], with Tx[2*k]
        and Tx[2*k+1] being the real and imaginary part of the kth entry.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            2 or less: no output.  returns silently without checking anything.
            3: fully check input, and print a short summary of its status
            4: as 3, but print first few entries of the input
            5: as 3, but print all of the input
            Default: 1
*/

//------------------------------------------------------------------------------
// umfpack_report_vector
//------------------------------------------------------------------------------

int umfpack_di_report_vector
(
    int32_t n,
    const double X [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_dl_report_vector
(
    int64_t n,
    const double X [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zi_report_vector
(
    int32_t n,
    const double Xx [ ], const double Xz [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zl_report_vector
(
    int64_t n,
    const double Xx [ ], const double Xz [ ],
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t n ;
    double *X, Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_report_vector (n, X, Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t n ;
    double *X, Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_report_vector (n, X, Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t n ;
    double *Xx, *Xz, Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_report_vector (n, Xx, Xz, Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t n ;
    double *Xx, *Xz, Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_report_vector (n, Xx, Xz, Control) ;

Purpose:

    Verifies and prints a dense vector.

Returns:

    UMFPACK_OK if Control [UMFPACK_PRL] <= 2 (the input is not checked).

    Otherwise:

    UMFPACK_OK if the vector is valid.
    UMFPACK_ERROR_argument_missing if X or Xx is missing.
    UMFPACK_ERROR_n_nonpositive if n <= 0.

Arguments:

    Int n ;             Input argument, not modified.

        X is a real or complex vector of size n.  Restriction: n > 0.

    double X [n] ;      Input argument, not modified.  For real versions.

        A real vector of size n.  X must not be (double *) NULL.

    double Xx [n or 2*n] ; Input argument, not modified.  For complex versions.
    double Xz [n or 0] ;   Input argument, not modified.  For complex versions.

        A complex vector of size n, in one of two storage formats.
        Xx must not be (double *) NULL.

        If Xz is not (double *) NULL, then Xx [i] is the real part of X (i) and
        Xz [i] is the imaginary part of X (i).  Both vectors are of length n.
        This is the "split" form of the complex vector X.

        If Xz is (double *) NULL, then Xx holds both real and imaginary parts,
        where Xx [2*i] is the real part of X (i) and Xx [2*i+1] is the imaginary
        part of X (i).  Xx is of length 2*n doubles.  If you have an ANSI C99
        compiler with the intrinsic double complex type, then Xx can be of
        type double complex in the calling routine and typecast to (double *)
        when passed to umfpack_*_report_vector (this is untested, however).
        This is the "merged" form of the complex vector X.

        Note that all complex routines in UMFPACK V4.4 and later use this same
        strategy for their complex arguments.  The split format is useful for
        MATLAB, which holds its real and imaginary parts in seperate arrays.
        The packed format is compatible with the intrinsic double complex
        type in ANSI C99, and is also compatible with SuperLU's method of
        storing complex matrices.  In Version 4.3, this routine was the only
        one that allowed for packed complex arguments.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            2 or less: no output.  returns silently without checking anything.
            3: fully check input, and print a short summary of its status
            4: as 3, but print first few entries of the input
            5: as 3, but print all of the input
            Default: 1
*/

//------------------------------------------------------------------------------
// umfpack_report_symbolic
//------------------------------------------------------------------------------

int umfpack_di_report_symbolic
(
    void *Symbolic,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_dl_report_symbolic
(
    void *Symbolic,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zi_report_symbolic
(
    void *Symbolic,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zl_report_symbolic
(
    void *Symbolic,
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_report_symbolic (Symbolic, Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_report_symbolic (Symbolic, Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_report_symbolic (Symbolic, Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Symbolic ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_report_symbolic (Symbolic, Control) ;

Purpose:

    Verifies and prints a Symbolic object.  This routine checks the object more
    carefully than the computational routines.  Normally, this check is not
    required, since umfpack_*_*symbolic either returns (void *) NULL, or a valid
    Symbolic object.  However, if you suspect that your own code has corrupted
    the Symbolic object (by overruning memory bounds, for example), then this
    routine might be able to detect a corrupted Symbolic object.  Since this is
    a complex object, not all such user-generated errors are guaranteed to be
    caught by this routine.

Returns:

    UMFPACK_OK if Control [UMFPACK_PRL] is <= 2 (no inputs are checked).

    Otherwise:

    UMFPACK_OK if the Symbolic object is valid.
    UMFPACK_ERROR_invalid_Symbolic_object if the Symbolic object is invalid.
    UMFPACK_ERROR_out_of_memory if out of memory.

Arguments:

    void *Symbolic ;                    Input argument, not modified.

        The Symbolic object, which holds the symbolic factorization computed by
        umfpack_*_*symbolic.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            2 or less: no output.  returns silently without checking anything.
            3: fully check input, and print a short summary of its status
            4: as 3, but print first few entries of the input
            5: as 3, but print all of the input
            Default: 1
*/

//------------------------------------------------------------------------------
// umfpack_report_numeric
//------------------------------------------------------------------------------

int umfpack_di_report_numeric
(
    void *Numeric,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_dl_report_numeric
(
    void *Numeric,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zi_report_numeric
(
    void *Numeric,
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zl_report_numeric
(
    void *Numeric,
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_report_numeric (Numeric, Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_report_numeric (Numeric, Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_report_numeric (Numeric, Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    void *Numeric ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_report_numeric (Numeric, Control) ;

Purpose:

    Verifies and prints a Numeric object (the LU factorization, both its pattern
    numerical values, and permutation vectors P and Q).  This routine checks the
    object more carefully than the computational routines.  Normally, this check
    is not required, since umfpack_*_numeric either returns (void *) NULL, or a
    valid Numeric object.  However, if you suspect that your own code has
    corrupted the Numeric object (by overruning memory bounds, for example),
    then this routine might be able to detect a corrupted Numeric object.  Since
    this is a complex object, not all such user-generated errors are guaranteed
    to be caught by this routine.

Returns:

    UMFPACK_OK if Control [UMFPACK_PRL] <= 2 (the input is not checked).

    Otherwise:

    UMFPACK_OK if the Numeric object is valid.
    UMFPACK_ERROR_invalid_Numeric_object if the Numeric object is invalid.
    UMFPACK_ERROR_out_of_memory if out of memory.

Arguments:

    void *Numeric ;                     Input argument, not modified.

        The Numeric object, which holds the numeric factorization computed by
        umfpack_*_numeric.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            2 or less: no output.  returns silently without checking anything.
            3: fully check input, and print a short summary of its status
            4: as 3, but print first few entries of the input
            5: as 3, but print all of the input
            Default: 1
*/

//------------------------------------------------------------------------------
// umfpack_report_perm
//------------------------------------------------------------------------------

int umfpack_di_report_perm
(
    int32_t np,
    const int32_t Perm [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_dl_report_perm
(
    int64_t np,
    const int64_t Perm [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zi_report_perm
(
    int32_t np,
    const int32_t Perm [ ],
    const double Control [UMFPACK_CONTROL]
) ;

int umfpack_zl_report_perm
(
    int64_t np,
    const int64_t Perm [ ],
    const double Control [UMFPACK_CONTROL]
) ;

/*
double int32_t Syntax:

    #include "umfpack.h"
    int32_t np, *Perm ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_di_report_perm (np, Perm, Control) ;

double int64_t Syntax:

    #include "umfpack.h"
    int64_t np, *Perm ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_dl_report_perm (np, Perm, Control) ;

complex int32_t Syntax:

    #include "umfpack.h"
    int32_t np, *Perm ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_zi_report_perm (np, Perm, Control) ;

complex int64_t Syntax:

    #include "umfpack.h"
    int64_t np, *Perm ;
    double Control [UMFPACK_CONTROL] ;
    int status = umfpack_zl_report_perm (np, Perm, Control) ;

Purpose:

    Verifies and prints a permutation vector.

Returns:

    UMFPACK_OK if Control [UMFPACK_PRL] <= 2 (the input is not checked).

    Otherwise:
    UMFPACK_OK if the permutation vector is valid (this includes that case
        when Perm is (Int *) NULL, which is not an error condition).
    UMFPACK_ERROR_n_nonpositive if np <= 0.
    UMFPACK_ERROR_out_of_memory if out of memory.
    UMFPACK_ERROR_invalid_permutation if Perm is not a valid permutation vector.

Arguments:

    Int np ;            Input argument, not modified.

        Perm is an integer vector of size np.  Restriction: np > 0.

    Int Perm [np] ;     Input argument, not modified.

        A permutation vector of size np.  If Perm is not present (an (Int *)
        NULL pointer), then it is assumed to be the identity permutation.  This
        is consistent with its use as an input argument to umfpack_*_qsymbolic,
        and is not an error condition.  If Perm is present, the entries in Perm
        must range between 0 and np-1, and no duplicates may exist.

    double Control [UMFPACK_CONTROL] ;  Input argument, not modified.

        If a (double *) NULL pointer is passed, then the default control
        settings are used.  Otherwise, the settings are determined from the
        Control array.  See umfpack_*_defaults on how to fill the Control
        array with the default settings.  If Control contains NaN's, the
        defaults are used.  The following Control parameters are used:

        Control [UMFPACK_PRL]:  printing level.

            2 or less: no output.  returns silently without checking anything.
            3: fully check input, and print a short summary of its status
            4: as 3, but print first few entries of the input
            5: as 3, but print all of the input
            Default: 1
*/

//==============================================================================
//==== Utility Routines ========================================================
//==============================================================================

//------------------------------------------------------------------------------
// umfpack_timer
//------------------------------------------------------------------------------

double umfpack_timer ( void ) ;

/*
Syntax (for all versions: di, dl, zi, and zl):

    #include "umfpack.h"
    double t ;
    t = umfpack_timer ( ) ;

Purpose:

    Returns the current wall clock time on POSIX C 1993 systems.

Arguments:

    None.
*/

//------------------------------------------------------------------------------
// umfpack_tic and umfpack_toc
//------------------------------------------------------------------------------

void umfpack_tic (double stats [2]) ;

void umfpack_toc (double stats [2]) ;

/*
Syntax (for all versions: di, dl, zi, and zl):

    #include "umfpack.h"
    double stats [2] ;
    umfpack_tic (stats) ;
    ...
    umfpack_toc (stats) ;

Purpose:

    umfpack_tic returns the wall clock time.
    umfpack_toc returns the wall clock time since the
    last call to umfpack_tic with the same stats array.

    Typical usage:

        umfpack_tic (stats) ;
        ... do some work ...
        umfpack_toc (stats) ;

    then stats [0] contains the elapsed wall clock time in seconds between
    umfpack_tic and umfpack_toc.

Arguments:

    double stats [2]:

        stats [0]:  wall clock time, in seconds
        stats [1]:  (same; was CPU time in prior versions)
*/

#ifdef __cplusplus
}
#endif

#endif /* UMFPACK_H */
