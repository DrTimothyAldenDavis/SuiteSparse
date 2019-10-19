/* ========================================================================== */
/* === klu include file ================================================= */
/* ========================================================================== */

/* Include file for user programs that call klu_* routines */

#ifndef _KLU_H
#define _KLU_H

/* make it easy for C++ programs to include KLU */
#ifdef __cplusplus
extern "C" {
#endif

#include "amd.h"
#include "colamd.h"
#include "btf.h"

/* -------------------------------------------------------------------------- */
/* Symbolic object - contains the pre-ordering computed by klu_analyze */
/* -------------------------------------------------------------------------- */

typedef struct
{
    /* A (P,Q) is in upper block triangular form.  The kth block goes from
     * row/col index R [k] to R [k+1]-1.  The estimated number of nonzeros
     * in the L factor of the kth block is Lnz [k]. 
     */

    /* only computed if the AMD ordering is chosen: */
    double symmetry ;	/* symmetry of largest block */
    double est_flops ;	/* est. factorization flop count */
    double lnz, unz ;	/* estimated nz in L and U, including diagonals */
    double *Lnz ;	/* size n, but only Lnz [0..nblocks-1] is used */

    /* computed for all orderings: */
    int
	n,		/* input matrix A is n-by-n */
	nz,		/* # entries in input matrix */
	*P, 		/* size n */
	*Q,		/* size n */
	*R,		/* size n+1, but only R [0..nblocks] is used */
	nzoff,		/* nz in off-diagonal blocks */
	nblocks,	/* number of blocks */
	maxblock,	/* size of largest block */
	ordering,	/* ordering used (AMD, COLAMD, or GIVEN) */
	do_btf ;	/* whether or not BTF preordering was requested */

    /* only computed if BTF preordering requested */
    int structural_rank ;   /* 0 to n-1 if the matrix is structurally rank
			* deficient.  -1 if not computed.  n if the matrix has
			* full structural rank */

} klu_symbolic ;

/* -------------------------------------------------------------------------- */
/* Numeric object - contains the factors computed by klu_factor */
/* -------------------------------------------------------------------------- */

typedef struct
{
    /* LU factors of each block, the pivot row permutation, and the
     * entries in the off-diagonal blocks */

    int nblocks ;	/* number of diagonal blocks */
    int lnz ;		/* actual nz in L, including diagonal */
    int unz ;		/* actual nz in U, including diagonal */
    int max_lnz_block ;	/* max actual nz in L in any one block, incl. diag */
    int max_unz_block ;	/* max actual nz in U in any one block, incl. diag */
    int *Pnum ;		/* size n. final pivot permutation */
    int *Pinv ;		/* size n. inverse of final pivot permutation */

    /* LU factors of each block */
    int **Lbip ;	/* TODO describe */
    int **Ubip ;	/* TODO describe */
    int **Lblen ;	/* TODO describe */
    int **Ublen ;	/* TODO describe */
    void **LUbx ;	/* L and U indices and entries (excl. diagonal of U) */
    void **Udiag ;	/* diagonal of U */
    void *Singleton ;	/* singleton values */

    double *Rs ;	/* size n.  row scaling factors, NULL if no scaling */

    /* permanent workspace for factorization and solve */
    size_t worksize ;	/* size (in bytes) of Work */
    void *Work ;	/* workspace */
    void *Xwork ;	/* alias into Numeric->Work */
    int *Iwork ;	/* alias into Numeric->Work */

    /* off-diagonal entries */
    int *Offp ;		/* TODO describe */
    int *Offi ;		/* TODO describe */
    void *Offx ;	/* TODO describe */

} klu_numeric ;

/* -------------------------------------------------------------------------- */
/* KLU control parameters and statistics */
/* -------------------------------------------------------------------------- */

/* return values of klu */
#define KLU_OK 0
#define KLU_SINGULAR (1)	    /* status > 0 is a warning, not an error */
#define KLU_OUT_OF_MEMORY (-2)
#define KLU_INVALID (-3)
#define KLU_TOO_LARGE (-4)	    /* integer overflow has occured */

typedef struct
{

    /* ---------------------------------------------------------------------- */
    /* parameters */
    /* ---------------------------------------------------------------------- */

    double tol ;		/* pivot tolerance for diagonal preference */
    double growth ;		/* realloc growth size */
    double initmem_amd ;	/* init. memory size with AMD: c*nnz(L) + n */
    double initmem ;		/* init. memory size: c*nnz(A) + n */
    int btf ;			/* use BTF pre-ordering, or not */
    int ordering ;		/* 0: AMD, 1: COLAMD, 2: user P and Q,
				 * 3: user function */
    int scale ;			/* row scaling: -1: none (and no error check),
				 * 0: none, 1: sum, 2: max */

    /* memory management routines */
    void *(*malloc_memory) (size_t) ;		/* pointer to malloc */
    void *(*realloc_memory) (void *, size_t) ;  /* pointer to realloc */
    void (*free_memory) (void *) ;		/* pointer to free */
    void *(*calloc_memory) (size_t, size_t) ;	/* pointer to calloc */

    /* pointer to user ordering function */
    int (*user_order) (int, int *, int *, int *, void *) ;

    /* pointer to user data, passed unchanged as the last parameter to the
     * user ordering function (optional, the user function need not use this
     * information). */
    void *user_data ;

    int halt_if_singular ;	/* how to handle a singular matrix:
	* FALSE: keep going.  Return a Numeric object with a zero U(k,k).  A
	*   divide-by-zero may occur when computing L(:,k).  The Numeric object
	*   can be passed to klu_solve (a divide-by-zero will occur).  It can
	*   also be safely passed to klu_refactor.
	* TRUE: stop quickly.  klu_factor will free the partially-constructed
	*   Numeric object.  klu_refactor will not free it, but will leave the
	*   numerical values only partially defined.  This is the default. */

    /* ---------------------------------------------------------------------- */
    /* statistics */
    /* ---------------------------------------------------------------------- */

    int status ;	        /* KLU_OK if OK, < 0 if error */
    int nrealloc ;		/* # of reallocations of L and U */

    int structural_rank ;	/* 0 to n-1 if the matrix is structurally rank
	* deficient (as determined by maxtrans).  -1 if not computed.  n if the
	* matrix has full structural rank.  This is computed by klu_analyze
	* if a BTF preordering is requested. */

    int numerical_rank ;	/* First k for which a zero U(k,k) was found,
	* if the matrix was singular (in the range 0 to n-1).  n if the matrix
	* has full rank. This is not a true rank-estimation.  It just reports
	* where the first zero pivot was found.  -1 if not computed.
	* Computed by klu_factor and klu_refactor. */

    int singular_col ;		/* n if the matrix is not singular.  If in the
	* range 0 to n-1, this is the column index of the original matrix A that
	* corresponds to the column of U that contains a zero diagonal entry.
	* -1 if not computed.  Computed by klu_factor and klu_refactor. */

    int noffdiag ;	/* # of off-diagonal pivots, -1 if not computed */

    /* statistic determined in klu_flops: */
    double flops ;	/* actual factorization flop count */

} klu_common ;

/* -------------------------------------------------------------------------- */
/* klu_analyze:  pre-orderings and analyzes a matrix with BTF and AMD */
/* -------------------------------------------------------------------------- */

/* Order the matrix with BTF (or not), then order each block with AMD, COLAMD,
 * a natural ordering, or with a user-provided ordering function */

klu_symbolic *klu_analyze
(
    /* inputs, not modified */
    int n,		/* A is n-by-n */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_analyze_given: analyzes a matrix using given P and Q */
/* -------------------------------------------------------------------------- */

/* Order the matrix with BTF (or not), then use natural or given ordering
 * Puser and Quser on the blocks.  Puser and Quser are interpretted as identity
 * if NULL. */

klu_symbolic *klu_analyze_given
(
    /* inputs, not modified */
    int n,		/* A is n-by-n */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    int Puser [ ],	/* size n, user's row permutation (may be NULL) */
    int Quser [ ],	/* size n, user's column permutation (may be NULL) */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_factor:  factors a matrix using the klu_analyze results */
/* -------------------------------------------------------------------------- */

klu_numeric *klu_factor	/* returns KLU_OK if OK, < 0 if error */
(
    /* inputs, not modified */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_common *Common
) ;

klu_numeric *klu_z_factor      /* returns KLU_OK if OK, < 0 if error */
(
     /* inputs, not modified */
     int Ap [ ],         /* size n+1, column pointers */
     int Ai [ ],         /* size nz, row indices */
     double Ax [ ],
     klu_symbolic *Symbolic,
     klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_free_symbolic: destroys the Symbolic object */
/* -------------------------------------------------------------------------- */

int klu_free_symbolic
(
    klu_symbolic **Symbolic,
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_free_numeric: destroys the Numeric object */
/* -------------------------------------------------------------------------- */

int klu_free_numeric
(
    klu_numeric **Numeric,
    klu_common *Common
) ;

int klu_z_free_numeric
(
     klu_numeric **Numeric,
     klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_solve: solves Ax=b using the Symbolic and Numeric objects */
/* -------------------------------------------------------------------------- */

int klu_solve
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    int ldim,		    /* leading dimension of B */
    int nrhs,		    /* number of right-hand-sides */

    /* right-hand-side on input, overwritten with solution to Ax=b on output */
    double B [ ],
    klu_common *Common
) ;

int klu_z_solve
(
     /* inputs, not modified */
     klu_symbolic *Symbolic,
     klu_numeric *Numeric,
     int ldim,               /* leading dimension of B */
     int nrhs,               /* number of right-hand-sides */

     /* right-hand-side on input, overwritten with solution to Ax=b on output */
     double B [ ],
     klu_common *Common
) ;


/* -------------------------------------------------------------------------- */
/* klu_tsolve: solves A'x=b using the Symbolic and Numeric objects */
/* -------------------------------------------------------------------------- */

int klu_tsolve
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    int ldim,		    /* leading dimension of B */
    int nrhs,		    /* number of right-hand-sides */

    /* right-hand-side on input, overwritten with solution to Ax=b on output */
    double B [ ],
    klu_common *Common
) ;

int klu_z_tsolve
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    int ldim,		    /* leading dimension of B */
    int nrhs,		    /* number of right-hand-sides */

    /* right-hand-side on input, overwritten with solution to Ax=b on output */
    double B [ ],
    int conj_solve,	    /* TRUE: conjugate solve, FALSE: solve A.'x=b */
    klu_common *Common
     
) ;


/* -------------------------------------------------------------------------- */
/* klu_refactor: refactorizes matrix with same ordering as klu_factor */
/* -------------------------------------------------------------------------- */

int klu_refactor	    /* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    double Ax [ ],
    klu_symbolic *Symbolic,
    /* input, and numerical values modified on output */
    klu_numeric *Numeric,
    klu_common *Common
) ;

int klu_z_refactor	    /* return TRUE if successful, FALSE otherwise */
(
     /* inputs, not modified */
     int Ap [ ],         /* size n+1, column pointers */
     int Ai [ ],         /* size nz, row indices */
     double Ax [ ],
     klu_symbolic *Symbolic,
     /* input, and numerical values modified on output */
     klu_numeric *Numeric,
     klu_common *Common
) ;


/* -------------------------------------------------------------------------- */
/* klu_sort: sorts the columns of the LU factorization */
/* -------------------------------------------------------------------------- */

int klu_sort
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    /* input/output */
    klu_numeric *Numeric,
    klu_common *Common
) ;

int klu_z_sort
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    /* input/output */
    klu_numeric *Numeric,
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_flops: determines # of flops performed in numeric factorzation */
/* -------------------------------------------------------------------------- */

int klu_flops
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    /* input/output */
    klu_common *Common
) ;

int klu_z_flops
(
    /* inputs, not modified */
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    /* input/output */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_defaults: sets default control parameters */
/* -------------------------------------------------------------------------- */

int klu_defaults
(
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_growth */
/* -------------------------------------------------------------------------- */

int klu_growth
(
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    double *growth,    /* reciprocal pivot growth */
    klu_common *Common
) ;
	
int klu_z_growth
(
    int Ap [ ],
    int Ai [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    double *growth,    /* reciprocal pivot growth */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_condest */
/* -------------------------------------------------------------------------- */

int klu_condest
(
    int Ap [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    double *condest,	    /* Output parameter : condition number */
    klu_common *Common
) ;

int klu_z_condest
(
    int Ap [ ],
    double Ax [ ],
    klu_symbolic *Symbolic,
    klu_numeric *Numeric,
    double *condest,	    /* Output parameter : condition number */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_scale */
/* -------------------------------------------------------------------------- */

int klu_scale		/* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int scale,		/* <0: none, no error check; 0: none, 1: sum, 2: max */
    int n,
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    double Ax [ ],
    /* outputs, not defined on input */
    double Rs [ ],
    /* workspace, not defined on input or output */
    int W [ ],		/* size n, can be NULL */
    klu_common *Common
) ;

int klu_z_scale		/* return TRUE if successful, FALSE otherwise */
(
    /* inputs, not modified */
    int scale,		/* <0: none, no error check; 0: none, 1: sum, 2: max */
    int n,
    int Ap [ ],		/* size n+1, column pointers */
    int Ai [ ],		/* size nz, row indices */
    double Ax [ ],
    /* outputs, not defined on input */
    double Rs [ ],
    /* workspace, not defined on input or output */
    int W [ ],		/* size n, can be NULL */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_extract  */
/* -------------------------------------------------------------------------- */

int klu_extract	    /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* L */
    int *Lp,	    /* size n+1 */
    int *Li,	    /* size nnz(L) */
    double *Lx,	    /* size nnz(L) */

    /* U */
    int *Up,	    /* size n+1 */
    int *Ui,	    /* size nnz(U) */
    double *Ux,	    /* size nnz(U) */

    /* F */
    int *Fp,	    /* size n+1 */
    int *Fi,	    /* size nnz(F) */
    double *Fx,	    /* size nnz(F) */

    /* P, row permutation */
    int *P,	    /* size n */

    /* Q, column permutation */
    int *Q,	    /* size n */

    /* Rs, scale factors */
    double *Rs,	    /* size n */

    /* R, block boundaries */
    int *R	    /* size nblocks+1 */
) ;

int klu_z_extract	    /* returns TRUE if successful, FALSE otherwise */
(
    /* inputs: */
    klu_numeric *Numeric,
    klu_symbolic *Symbolic,

    /* outputs, all of which must be allocated on input */

    /* L */
    int *Lp,	    /* size n+1 */
    int *Li,	    /* size nnz(L) */
    double *Lx,	    /* size nnz(L) */
#ifdef COMPLEX
    double *Lz,	    /* size nnz(L) for the complex case, ignored if real */
#endif

    /* U */
    int *Up,	    /* size n+1 */
    int *Ui,	    /* size nnz(U) */
    double *Ux,	    /* size nnz(U) */
#ifdef COMPLEX
    double *Uz,	    /* size nnz(U) for the complex case, ignored if real */
#endif

    /* F */
    int *Fp,	    /* size n+1 */
    int *Fi,	    /* size nnz(F) */
    double *Fx,	    /* size nnz(F) */
#ifdef COMPLEX
    double *Fz,	    /* size nnz(F) for the complex case, ignored if real */
#endif

    /* P, row permutation */
    int *P,	    /* size n */

    /* Q, column permutation */
    int *Q,	    /* size n */

    /* Rs, scale factors */
    double *Rs,	    /* size n */

    /* R, block boundaries */
    int *R	    /* size nblocks+1 */
) ;

/* -------------------------------------------------------------------------- */
/* KLU memory management routines */
/* -------------------------------------------------------------------------- */

void *klu_malloc	/* returns pointer to the newly malloc'd block */
(
    /* ---- input ---- */
    size_t n,		/* number of items */
    size_t size,	/* size of each item */
    /* --------------- */
    klu_common *Common
) ;

void *klu_free		/* always returns NULL */
(
    /* ---- in/out --- */
    void *p,		/* block of memory to free */
    /* --------------- */
    klu_common *Common
) ;

void *klu_realloc	/* returns pointer to reallocated block */
(
    /* ---- input ---- */
    size_t nnew,	/* requested # of items in reallocated block */
    size_t size,	/* size of each item */
    /* ---- in/out --- */
    void *p,		/* block of memory to realloc */
    /* --------------- */
    klu_common *Common
) ;

/* -------------------------------------------------------------------------- */
/* klu_rcond: compute min(abs(diag(U))) / max(abs(diag(U))) */
/* -------------------------------------------------------------------------- */

int klu_rcond
(
    klu_symbolic *Symbolic,	    /* input, not modified */
    klu_numeric *Numeric,	    /* input, not modified */
    double *rcond,		    /* output (pointer to a scalar) */
    klu_common *Common
) ;

int klu_z_rcond
(
    klu_symbolic *Symbolic,	    /* input, not modified */
    klu_numeric *Numeric,	    /* input, not modified */
    double *rcond,		    /* output (pointer to a scalar) */
    klu_common *Common
) ;

/* ========================================================================== */
/* === KLU version ========================================================== */
/* ========================================================================== */

/* All versions of KLU include these definitions.
 * As an example, to test if the version you are using is 1.2 or later:
 *
 *	if (KLU_VERSION >= KLU_VERSION_CODE (1,2)) ...
 *
 * This also works during compile-time:
 *
 *	#if (KLU >= KLU_VERSION_CODE (1,2))
 *	    printf ("This is version 1.2 or later\n") ;
 *	#else
 *	    printf ("This is an early version\n") ;
 *	#endif
 */

#define KLU_DATE "May 23, 2006"
#define KLU_VERSION_CODE(main,sub) ((main) * 1000 + (sub))
#define KLU_MAIN_VERSION 0
#define KLU_SUB_VERSION 10
#define KLU_VERSION KLU_VERSION_CODE(KLU_MAIN_VERSION,KLU_SUB_VERSION)

#ifdef __cplusplus
}
#endif
#endif
