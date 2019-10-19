/* ========================================================================== */
/* === KLU/Include/klu_internal.h =========================================== */
/* ========================================================================== */

/* For internal use in KLU routines only, not for user programs */

#ifndef _KLU_INTERNAL_H
#define _KLU_INTERNAL_H

#include "klu.h" 
#include "btf.h"
#include "klu_version.h"

int KLU_kernel
(
    /* input, not modified */
    int n,		/* A is n-by-n */
    int Ap [ ],		/* size n+1, column pointers for A */
    int Ai [ ],		/* size nz = Ap [n], row indices for A */
    Entry Ax [ ],	/* size nz, values of A */
    int Q [ ],		/* size n, optional input permutation */
    size_t lusize,     /* initial size of LU, final size is Uxp[n] */

    /* output, not defined on input */
    int Pinv [ ],	/* size n */
    int P [ ],		/* size n */
    Unit **p_LU,	/* size lusize on input, size Uxp[n] on output*/
    Unit Udiag [ ],	/* size n, diagonal of U */
    int Llen [ ],	/* size n, column length of L */
    int Ulen [ ],	/* size n, column length of U */
    int Lip [ ],	/* size n+1 */
    int Uip [ ],	/* size n+1 */
    int *lnz,		/* size of L */
    int *unz,		/* size of U */

    /* workspace, not defined on input */

    Entry X [ ],   /* size n, zero on output */

    /* workspace, not defined on input or output */
    int Stack [ ],  /* size n */
    int Flag [ ],   /* size n */
    int adj_pos [ ],	/* size n */
    
    /* workspace for pruning only */
    int Lpend [ ],	/* size n workspace */


    /* ---- the following are only used in the BTF case --- */
    /* inputs, not modified on output */
    int no_btf,
    int k1,	    	/* the block of A is from k1 to k2-1 */
    int PSinv [ ],  	/* inverse of P from symbolic factorization */
    double Rs [ ],  	/* scale factors for A */

    /* inputs, modified on output */
    int Offp [ ],   /* off-diagonal matrix (modified by this routine) */
    int Offi [ ],
    Entry Offx [ ],
    klu_common *Common	/* the control input/output structure */
) ;

/* ========================================================================== */
/* make sure debugging and printing is turned off */

#ifndef NDEBUG
#define NDEBUG
#endif
#ifndef NPRINT
#define NPRINT
#endif

/* To enable debugging and assertions, uncomment this line:
 #undef NDEBUG
 */

/* To enable diagnostic printing, uncomment this line:
 #undef NPRINT
 */

/* ========================================================================== */

#include <stdio.h>
#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <math.h>

#define ASSERT(a) assert(a)

#define SCALAR_IS_NAN(x) ((x) != (x))

/* true if an integer (stored in double x) would overflow (or if x is NaN) */
#define INT_OVERFLOW(x) ((!((x) * (1.0+1e-8) <= (double) INT_MAX)) \
			|| SCALAR_IS_NAN (x))

#undef TRUE
#undef FALSE
#undef MAX
#undef MIN
#undef PRINTF
#undef FLIP

#ifndef NPRINT
#define PRINTF(s) { printf s ; } ;
#else
#define PRINTF(s)
#endif

#define TRUE 1
#define FALSE 0
#define MAX(a,b) (((a) > (b)) ?  (a) : (b))
#define MIN(a,b) (((a) < (b)) ?  (a) : (b))

/* FLIP is a "negation about -1", and is used to mark an integer i that is
 * normally non-negative.  FLIP (EMPTY) is EMPTY.  FLIP of a number > EMPTY
 * is negative, and FLIP of a number < EMTPY is positive.  FLIP (FLIP (i)) = i
 * for all integers i.  UNFLIP (i) is >= EMPTY. */
#define EMPTY (-1)
#define FLIP(i) (-(i)-2)
#define UNFLIP(i) (((i) < EMPTY) ? FLIP (i) : (i))


int KLU_kernel_factor	/* returns 0 if OK, negative if error */
(
    /* inputs, not modified */
    int n,	    /* A is n-by-n. n must be > 0. */
    int Ap [ ],	    /* size n+1, column pointers for A */
    int Ai [ ],	    /* size nz = Ap [n], row indices for A */
    Entry Ax [ ],   /* size nz, values of A */
    int Q [ ],	    /* size n, optional column permutation */
    double Lsize,   /* initial size of L and U */

    /* outputs, not defined on input */
    Unit **p_LU,	/* row indices and values of L and U */
    Unit Udiag [ ],	/* size n, diagonal of U */
    int Llen [ ],	/* size n, column length of L */
    int Ulen [ ],	/* size n, column length of U */
    int Lip [ ],	/* size n+1, column pointers of L */
    int Uip [ ],	/* size n+1, column pointers of U */
    int P [ ],	        /* row permutation, size n */
    int *lnz,	   	/* size of L */
    int *unz,	    	/* size of U */

    /* workspace, undefined on input */
    Entry *X,	    /* size n entries.  Zero on output */
    int *Work,	    /* size 5n int's */

    /* ---- the following are only used in the BTF case --- */

    /* inputs, not modified on output */
    int k1,	    	/* the block of A is from k1 to k2-1 */
    int PSinv [ ],  	/* inverse of P from symbolic factorization */
    double Rs [ ],  	/* scale factors for A */

    /* inputs, modified on output */
    int Offp [ ],   /* off-diagonal matrix (modified by this routine) */
    int Offi [ ],
    Entry Offx [ ],
    klu_common *Common	/* the control input/output structure */
) ;

void KLU_lsolve
(
    /* inputs, not modified: */
    int n,
    int Lp [ ],
    int Li [ ],
    Unit LU [ ],
    int nrhs,
    /* right-hand-side on input, solution to Lx=b on output */
    Entry X [ ]
) ;

void KLU_ltsolve
(
    /* inputs, not modified: */
    int n,
    int Lp [ ],
    int Li [ ],
    Unit LU [ ],
    int nrhs,
#ifdef COMPLEX
    int conj_solve,
#endif
    /* right-hand-side on input, solution to L'x=b on output */
    Entry X [ ]
) ;


void KLU_usolve
(
    /* inputs, not modified: */
    int n,
    int Up [ ],
    int Ui [ ],
    Unit LU [ ],
    Unit Udiag [ ],
    int nrhs,
    /* right-hand-side on input, solution to Ux=b on output */
    Entry X [ ]
) ;

void KLU_utsolve
(
    /* inputs, not modified: */
    int n,
    int Up [ ],
    int Ui [ ],
    Unit LU [ ],
    Unit Udiag [ ],
    int nrhs,
#ifdef COMPLEX
    int conj_solve,
#endif
    /* right-hand-side on input, solution to U'x=b on output */
    Entry X [ ]
) ;

int KLU_valid 
(
    int n, 
    int Ap [ ], 
    int Ai [ ], 
    Entry Ax [ ]
) ;

int KLU_valid_LU 
(
    int n, 
    int flag_test_start_ptr, 
    int Xip [ ],
    int Xlen [ ],  
    Unit LU [ ]
);

size_t klu_add_size_t (size_t a, size_t b, int *ok) ;

size_t klu_mult_size_t (size_t a, size_t k, int *ok) ;

#endif
