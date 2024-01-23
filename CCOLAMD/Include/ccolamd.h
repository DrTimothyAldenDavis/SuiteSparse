//------------------------------------------------------------------------------
// CCOLAMD/Include/ccolamd.h:  constrained column approx. min. degree ordering
//------------------------------------------------------------------------------

// CCOLAMD, Copyright (c) 1996-2024, Timothy A. Davis, Sivasankaran
// Rajamanickam, and Stefan Larimore.  All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/*
 *  You must include this file (ccolamd.h) in any routine that uses ccolamd,
 *  csymamd, or the related macros and definitions.
 */

#ifndef CCOLAMD_H
#define CCOLAMD_H

#include "SuiteSparse_config.h"

/* ========================================================================== */
/* === CCOLAMD version ====================================================== */
/* ========================================================================== */

/* All versions of CCOLAMD will include the following definitions.
 * As an example, to test if the version you are using is 1.3 or later:
 *
 *	if (CCOLAMD_VERSION >= CCOLAMD_VERSION_CODE (1,3)) ...
 *
 * This also works during compile-time:
 *
 *	#if CCOLAMD_VERSION >= CCOLAMD_VERSION_CODE (1,3)
 *	    printf ("This is version 1.3 or later\n") ;
 *	#else
 *	    printf ("This is an early version\n") ;
 *	#endif
 */

#define CCOLAMD_DATE "Jan 20, 2024"
#define CCOLAMD_MAIN_VERSION   3
#define CCOLAMD_SUB_VERSION    3
#define CCOLAMD_SUBSUB_VERSION 2

#define CCOLAMD_VERSION_CODE(main,sub) SUITESPARSE_VER_CODE(main,sub)
#define CCOLAMD_VERSION CCOLAMD_VERSION_CODE(3,3)

#define CCOLAMD__VERSION SUITESPARSE__VERCODE(3,3,2)
#if !defined (SUITESPARSE__VERSION) || \
    (SUITESPARSE__VERSION < SUITESPARSE__VERCODE(7,6,0))
#error "CCOLAMD 3.3.2 requires SuiteSparse_config 7.6.0 or later"
#endif

/* ========================================================================== */
/* === Knob and statistics definitions ====================================== */
/* ========================================================================== */

/* size of the knobs [ ] array.  Only knobs [0..3] are currently used. */
#define CCOLAMD_KNOBS 20

/* number of output statistics.  Only stats [0..10] are currently used. */
#define CCOLAMD_STATS 20

/* knobs [0] and stats [0]: dense row knob and output statistic. */
#define CCOLAMD_DENSE_ROW 0

/* knobs [1] and stats [1]: dense column knob and output statistic. */
#define CCOLAMD_DENSE_COL 1

/* knobs [2]: aggressive absorption option */
#define CCOLAMD_AGGRESSIVE 2

/* knobs [3]: LU or Cholesky factorization option */
#define CCOLAMD_LU 3

/* stats [2]: memory defragmentation count output statistic */
#define CCOLAMD_DEFRAG_COUNT 2

/* stats [3]: ccolamd status:  zero OK, > 0 warning or notice, < 0 error */
#define CCOLAMD_STATUS 3

/* stats [4..6]: error info, or info on jumbled columns */ 
#define CCOLAMD_INFO1 4
#define CCOLAMD_INFO2 5
#define CCOLAMD_INFO3 6

/* stats [7]: number of originally empty rows */
#define CCOLAMD_EMPTY_ROW 7
/* stats [8]: number of originally empty cols */
#define CCOLAMD_EMPTY_COL 8
/* stats [9]: number of rows with entries only in dense cols */
#define CCOLAMD_NEWLY_EMPTY_ROW 9
/* stats [10]: number of cols with entries only in dense rows */
#define CCOLAMD_NEWLY_EMPTY_COL 10

/* error codes returned in stats [3]: */
#define CCOLAMD_OK				(0)
#define CCOLAMD_OK_BUT_JUMBLED			(1)
#define CCOLAMD_ERROR_A_not_present		(-1)
#define CCOLAMD_ERROR_p_not_present		(-2)
#define CCOLAMD_ERROR_nrow_negative		(-3)
#define CCOLAMD_ERROR_ncol_negative		(-4)
#define CCOLAMD_ERROR_nnz_negative		(-5)
#define CCOLAMD_ERROR_p0_nonzero		(-6)
#define CCOLAMD_ERROR_A_too_small		(-7)
#define CCOLAMD_ERROR_col_length_negative	(-8)
#define CCOLAMD_ERROR_row_index_out_of_bounds	(-9)
#define CCOLAMD_ERROR_out_of_memory		(-10)
#define CCOLAMD_ERROR_invalid_cmember		(-11)
#define CCOLAMD_ERROR_internal_error		(-999)

/* ========================================================================== */
/* === Prototypes of user-callable routines ================================= */
/* ========================================================================== */

/* make it easy for C++ programs to include CCOLAMD */
#ifdef __cplusplus
extern "C" {
#endif

size_t ccolamd_recommended	/* returns recommended value of Alen, */
				/* or 0 if input arguments are erroneous */
(
    int nnz,			/* nonzeros in A */
    int n_row,			/* number of rows in A */
    int n_col			/* number of columns in A */
) ;

size_t ccolamd_l_recommended	/* returns recommended value of Alen, */
				/* or 0 if input arguments are erroneous */
(
    int64_t nnz,		/* nonzeros in A */
    int64_t n_row,		/* number of rows in A */
    int64_t n_col		/* number of columns in A */
) ;

void ccolamd_set_defaults	/* sets default parameters */
(				/* knobs argument is modified on output */
    double knobs [CCOLAMD_KNOBS]	/* parameter settings for ccolamd */
) ;

void ccolamd_l_set_defaults	/* sets default parameters */
(				/* knobs argument is modified on output */
    double knobs [CCOLAMD_KNOBS]	/* parameter settings for ccolamd */
) ;

int ccolamd			/* returns (1) if successful, (0) otherwise*/
(				/* A and p arguments are modified on output */
    int n_row,			/* number of rows in A */
    int n_col,			/* number of columns in A */
    int Alen,			/* size of the array A */
    int A [ ],			/* row indices of A, of size Alen */
    int p [ ],			/* column pointers of A, of size n_col+1 */
    double knobs [CCOLAMD_KNOBS],/* parameter settings for ccolamd */
    int stats [CCOLAMD_STATS],	/* ccolamd output statistics and error codes */
    int cmember [ ]		/* Constraint set of A, of size n_col */
) ;

int ccolamd_l      /* as ccolamd w/ int64_t integers */
(
    int64_t n_row,
    int64_t n_col,
    int64_t Alen,
    int64_t A [ ],
    int64_t p [ ],
    double knobs [CCOLAMD_KNOBS],
    int64_t stats [CCOLAMD_STATS],
    int64_t cmember [ ]
) ;

int csymamd			/* return (1) if OK, (0) otherwise */
(
    int n,			/* number of rows and columns of A */
    int A [ ],			/* row indices of A */
    int p [ ],			/* column pointers of A */
    int perm [ ],		/* output permutation, size n_col+1 */
    double knobs [CCOLAMD_KNOBS],/* parameters (uses defaults if NULL) */
    int stats [CCOLAMD_STATS],	/* output statistics and error codes */
    void * (*allocate) (size_t, size_t), /* pointer to calloc (ANSI C) or */
				/* mxCalloc (for MATLAB mexFunction) */
    void (*release) (void *),	/* pointer to free (ANSI C) or */
    				/* mxFree (for MATLAB mexFunction) */
    int cmember [ ],		/* Constraint set of A */
    int stype			/* 0: use both parts, >0: upper, <0: lower */
) ;

int csymamd_l      /* as csymamd, w/ int64_t integers */
(
    int64_t n,
    int64_t A [ ],
    int64_t p [ ],
    int64_t perm [ ],
    double knobs [CCOLAMD_KNOBS],
    int64_t stats [CCOLAMD_STATS],
    void * (*allocate) (size_t, size_t),
    void (*release) (void *),
    int64_t cmember [ ],
    int64_t stype
) ;

void ccolamd_report
(
    int stats [CCOLAMD_STATS]
) ;

void ccolamd_l_report
(
    int64_t stats [CCOLAMD_STATS]
) ;

void csymamd_report
(
    int stats [CCOLAMD_STATS]
) ;

void csymamd_l_report
(
    int64_t stats [CCOLAMD_STATS]
) ;

void ccolamd_version (int version [3]) ;

/* ========================================================================== */
/* === Prototypes of "expert" routines ====================================== */
/* ========================================================================== */

/* These routines are meant to be used internally, or in a future version of
 * UMFPACK.  They appear here so that UMFPACK can use them, but they should not
 * be called directly by the user.
 */

int ccolamd2
(				/* A and p arguments are modified on output */
    int n_row,			/* number of rows in A */
    int n_col,			/* number of columns in A */
    int Alen,			/* size of the array A */
    int A [ ],			/* row indices of A, of size Alen */
    int p [ ],			/* column pointers of A, of size n_col+1 */
    double knobs [CCOLAMD_KNOBS],/* parameter settings for ccolamd */
    int stats [CCOLAMD_STATS],	/* ccolamd output statistics and error codes */
    /* each Front_ array is of size n_col+1: */
    int Front_npivcol [ ],	/* # pivot cols in each front */
    int Front_nrows [ ],	/* # of rows in each front (incl. pivot rows) */
    int Front_ncols [ ],	/* # of cols in each front (incl. pivot cols) */
    int Front_parent [ ],	/* parent of each front */
    int Front_cols [ ],		/* link list of pivot columns for each front */
    int *p_nfr,			/* total number of frontal matrices */
    int InFront [ ],		/* InFront [row] = f if row in front f */
    int cmember [ ]		/* Constraint set of A */
) ;

int ccolamd2_l     /* as ccolamd2, w/ int64_t integers */
(
    int64_t n_row,
    int64_t n_col,
    int64_t Alen,
    int64_t A [ ],
    int64_t p [ ],
    double knobs [CCOLAMD_KNOBS],
    int64_t stats [CCOLAMD_STATS],
    int64_t Front_npivcol [ ],
    int64_t Front_nrows [ ],
    int64_t Front_ncols [ ],
    int64_t Front_parent [ ],
    int64_t Front_cols [ ],
    int64_t *p_nfr,
    int64_t InFront [ ],
    int64_t cmember [ ]
) ;

void ccolamd_apply_order
(
    int Front [ ],
    const int Order [ ],
    int Temp [ ],
    int nn,
    int nfr
) ;

void ccolamd_l_apply_order
(
    int64_t Front [ ],
    const int64_t Order [ ],
    int64_t Temp [ ],
    int64_t nn,
    int64_t nfr
) ;

void ccolamd_fsize
(
    int nn,
    int MaxFsize [ ],
    int Fnrows [ ],
    int Fncols [ ],
    int Parent [ ],
    int Npiv [ ]
) ;

void ccolamd_l_fsize
(
    int64_t nn,
    int64_t MaxFsize [ ],
    int64_t Fnrows [ ],
    int64_t Fncols [ ],
    int64_t Parent [ ],
    int64_t Npiv [ ]
) ;

void ccolamd_postorder
(
    int nn,
    int Parent [ ],
    int Npiv [ ],
    int Fsize [ ],
    int Order [ ],
    int Child [ ],
    int Sibling [ ],
    int Stack [ ],
    int Front_cols [ ],
    int cmember [ ]
) ;

void ccolamd_l_postorder
(
    int64_t nn,
    int64_t Parent [ ],
    int64_t Npiv [ ],
    int64_t Fsize [ ],
    int64_t Order [ ],
    int64_t Child [ ],
    int64_t Sibling [ ],
    int64_t Stack [ ],
    int64_t Front_cols [ ],
    int64_t cmember [ ]
) ;

int ccolamd_post_tree
(
    int root,
    int k,
    int Child [ ],
    const int Sibling [ ],
    int Order [ ],
    int Stack [ ]
) ;

int64_t ccolamd_l_post_tree
(
    int64_t root,
    int64_t k,
    int64_t Child [ ],
    const int64_t Sibling [ ],
    int64_t Order [ ],
    int64_t Stack [ ]
) ;

#ifdef __cplusplus
}
#endif

#endif
