/* ========================================================================== */
/* === klu_defaults ========================================================= */
/* ========================================================================== */

/* Sets default parameters for KLU */

#include "klu_internal.h"

int klu_defaults
(
    klu_common *Common
)
{
    if (Common == NULL)
    {
	return (FALSE) ;
    }

    /* parameters */
    Common->tol = 0.001 ;	/* pivot tolerance for diagonal */
    Common->growth = 1.2;	/* realloc growth size */
    Common->initmem_amd = 1.2 ;	/* init. mem with AMD:  c*nnz(L) + n */
    Common->initmem = 10 ;	/* init. mem otherwise: c*nnz(A) + n */
    Common->btf = TRUE ;	/* use BTF pre-ordering, or not */
    Common->ordering = 0 ;	/* 0: AMD, 1: COLAMD, 2: user-provided P and Q,
				 * 3: user-provided function */
    Common->scale = -1 ;	/* scale: -1: none, and do not check for errors
				 * in the input matrix in klu_refactor.
				 * 0: none, but check for errors,
				 * 1: sum, 2: max */
    Common->halt_if_singular = TRUE ;

    /* memory management routines */
    Common->malloc_memory  = malloc ;
    Common->calloc_memory  = calloc ;
    Common->free_memory    = free ;
    Common->realloc_memory = realloc ;

    /* user ordering function and optional argument */
    Common->user_order = NULL ;
    Common->user_data = NULL ;

    /* statistics */
    Common->status = KLU_OK ;
    Common->nrealloc = 0 ;
    Common->structural_rank = EMPTY ;
    Common->numerical_rank = EMPTY ;
    Common->noffdiag = EMPTY ;

    return (TRUE) ;
}
