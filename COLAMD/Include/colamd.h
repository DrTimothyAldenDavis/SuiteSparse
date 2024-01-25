//------------------------------------------------------------------------------
// COLAMD/Include/colamd.h: include file for COLAMD
//------------------------------------------------------------------------------

// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,
// All Rights Reserved.
// SPDX-License-Identifier: BSD-3-clause

//------------------------------------------------------------------------------

/* COLAMD / SYMAMD include file

    You must include this file (colamd.h) in any routine that uses colamd,
    symamd, or the related macros and definitions.

    Authors:

        The authors of the code itself are Stefan I. Larimore and Timothy A.
        Davis (DrTimothyAldenDavis@gmail.com).  The algorithm was
        developed in collaboration with John Gilbert, Xerox PARC, and Esmond
        Ng, Oak Ridge National Laboratory.

    Acknowledgements:

        This work was supported by the National Science Foundation, under
        grants DMS-9504974 and DMS-9803599.

    Availability:

        The colamd/symamd library is available at http://www.suitesparse.com
        This file is required by the colamd.c, colamdmex.c, and symamdmex.c
        files, and by any C code that calls the routines whose prototypes are
        listed below, or that uses the colamd/symamd definitions listed below.

*/

#ifndef COLAMD_H
#define COLAMD_H

/* ========================================================================== */
/* === Include files ======================================================== */
/* ========================================================================== */

#include "SuiteSparse_config.h"

/* ========================================================================== */
/* === COLAMD version ======================================================= */
/* ========================================================================== */

/* COLAMD Version 2.4 and later will include the following definitions.
 * As an example, to test if the version you are using is 2.4 or later:
 *
 * #ifdef COLAMD_VERSION
 *     if (COLAMD_VERSION >= COLAMD_VERSION_CODE (2,4)) ...
 * #endif
 *
 * This also works during compile-time:
 *
 *  #if defined(COLAMD_VERSION) && (COLAMD_VERSION >= COLAMD_VERSION_CODE (2,4))
 *    printf ("This is version 2.4 or later\n") ;
 *  #else
 *    printf ("This is an early version\n") ;
 *  #endif
 *
 * Versions 2.3 and earlier of COLAMD do not include a #define'd version number.
 */

#define COLAMD_DATE "Jan 20, 2024"
#define COLAMD_MAIN_VERSION   3
#define COLAMD_SUB_VERSION    3
#define COLAMD_SUBSUB_VERSION 2

#define COLAMD_VERSION_CODE(main,sub) SUITESPARSE_VER_CODE(main,sub)
#define COLAMD_VERSION COLAMD_VERSION_CODE(3,3)

#define COLAMD__VERSION SUITESPARSE__VERCODE(3,3,2)
#if !defined (SUITESPARSE__VERSION) || \
    (SUITESPARSE__VERSION < SUITESPARSE__VERCODE(7,6,0))
#error "COLAMD 3.3.2 requires SuiteSparse_config 7.6.0 or later"
#endif

/* ========================================================================== */
/* === Knob and statistics definitions ====================================== */
/* ========================================================================== */

/* size of the knobs [ ] array.  Only knobs [0..1] are currently used. */
#define COLAMD_KNOBS 20

/* number of output statistics.  Only stats [0..6] are currently used. */
#define COLAMD_STATS 20

/* knobs [0] and stats [0]: dense row knob and output statistic. */
#define COLAMD_DENSE_ROW 0

/* knobs [1] and stats [1]: dense column knob and output statistic. */
#define COLAMD_DENSE_COL 1

/* knobs [2]: aggressive absorption */
#define COLAMD_AGGRESSIVE 2

/* stats [2]: memory defragmentation count output statistic */
#define COLAMD_DEFRAG_COUNT 2

/* stats [3]: colamd status:  zero OK, > 0 warning or notice, < 0 error */
#define COLAMD_STATUS 3

/* stats [4..6]: error info, or info on jumbled columns */ 
#define COLAMD_INFO1 4
#define COLAMD_INFO2 5
#define COLAMD_INFO3 6

/* error codes returned in stats [3]: */
#define COLAMD_OK                               (0)
#define COLAMD_OK_BUT_JUMBLED                   (1)
#define COLAMD_ERROR_A_not_present              (-1)
#define COLAMD_ERROR_p_not_present              (-2)
#define COLAMD_ERROR_nrow_negative              (-3)
#define COLAMD_ERROR_ncol_negative              (-4)
#define COLAMD_ERROR_nnz_negative               (-5)
#define COLAMD_ERROR_p0_nonzero                 (-6)
#define COLAMD_ERROR_A_too_small                (-7)
#define COLAMD_ERROR_col_length_negative        (-8)
#define COLAMD_ERROR_row_index_out_of_bounds    (-9)
#define COLAMD_ERROR_out_of_memory              (-10)
#define COLAMD_ERROR_internal_error             (-999)


/* ========================================================================== */
/* === Prototypes of user-callable routines ================================= */
/* ========================================================================== */

/* make it easy for C++ programs to include COLAMD */
#ifdef __cplusplus
extern "C" {
#endif

size_t colamd_recommended       /* returns recommended value of Alen, */
                                /* or 0 if input arguments are erroneous */
(
    int32_t nnz,                /* nonzeros in A */
    int32_t n_row,              /* number of rows in A */
    int32_t n_col               /* number of columns in A */
) ;

size_t colamd_l_recommended     /* returns recommended value of Alen, */
                                /* or 0 if input arguments are erroneous */
(
    int64_t nnz,                /* nonzeros in A */
    int64_t n_row,              /* number of rows in A */
    int64_t n_col               /* number of columns in A */
) ;

void colamd_set_defaults        /* sets default parameters */
(                               /* knobs argument is modified on output */
    double knobs [COLAMD_KNOBS] /* parameter settings for colamd */
) ;

void colamd_l_set_defaults      /* sets default parameters */
(                               /* knobs argument is modified on output */
    double knobs [COLAMD_KNOBS] /* parameter settings for colamd */
) ;

int colamd                      /* returns (1) if successful, (0) otherwise*/
(                               /* A and p arguments are modified on output */
    int32_t n_row,              /* number of rows in A */
    int32_t n_col,              /* number of columns in A */
    int32_t Alen,               /* size of the array A */
    int32_t A [],               /* row indices of A, of size Alen */
    int32_t p [],               /* column pointers of A, of size n_col+1 */
    double knobs [COLAMD_KNOBS],    /* parameter settings for colamd */
    int32_t stats [COLAMD_STATS]    /* colamd output stats and error codes */
) ;

int colamd_l                    /* returns (1) if successful, (0) otherwise*/
(                               /* A and p arguments are modified on output */
    int64_t n_row,              /* number of rows in A */
    int64_t n_col,              /* number of columns in A */
    int64_t Alen,               /* size of the array A */
    int64_t A [],               /* row indices of A, of size Alen */
    int64_t p [],               /* column pointers of A, of size n_col+1 */
    double knobs [COLAMD_KNOBS],    /* parameter settings for colamd */
    int64_t stats [COLAMD_STATS]    /* colamd output stats and error codes */
) ;

int symamd                              /* return (1) if OK, (0) otherwise */
(
    int32_t n,                          /* number of rows and columns of A */
    int32_t A [],                       /* row indices of A */
    int32_t p [],                       /* column pointers of A */
    int32_t perm [],                    /* output permutation, size n_col+1 */
    double knobs [COLAMD_KNOBS],        /* parameters (uses defaults if NULL) */
    int32_t stats [COLAMD_STATS],       /* output stats and error codes */
    void * (*allocate) (size_t, size_t),
                                        /* pointer to calloc (ANSI C) or */
                                        /* mxCalloc (for MATLAB mexFunction) */
    void (*release) (void *)
                                        /* pointer to free (ANSI C) or */
                                        /* mxFree (for MATLAB mexFunction) */
) ;

int symamd_l                            /* return (1) if OK, (0) otherwise */
(
    int64_t n,                          /* number of rows and columns of A */
    int64_t A [],                       /* row indices of A */
    int64_t p [],                       /* column pointers of A */
    int64_t perm [],                    /* output permutation, size n_col+1 */
    double knobs [COLAMD_KNOBS],        /* parameters (uses defaults if NULL) */
    int64_t stats [COLAMD_STATS],       /* output stats and error codes */
    void * (*allocate) (size_t, size_t),
                                        /* pointer to calloc (ANSI C) or */
                                        /* mxCalloc (for MATLAB mexFunction) */
    void (*release) (void *)
                                        /* pointer to free (ANSI C) or */
                                        /* mxFree (for MATLAB mexFunction) */
) ;

void colamd_report
(
    int32_t stats [COLAMD_STATS]
) ;

void colamd_l_report
(
    int64_t stats [COLAMD_STATS]
) ;

void symamd_report
(
    int32_t stats [COLAMD_STATS]
) ;

void symamd_l_report
(
    int64_t stats [COLAMD_STATS]
) ;

void colamd_version (int version [3]) ;

#ifdef __cplusplus
}
#endif

#endif /* COLAMD_H */
