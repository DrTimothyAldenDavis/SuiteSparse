/* ========================================================================== */
/* === Cholesky/t_cholmod_solve ============================================= */
/* ========================================================================== */

/* -----------------------------------------------------------------------------
 * CHOLMOD/Cholesky Module.  Copyright (C) 2005-2006, Timothy A. Davis
 * The CHOLMOD/Cholesky Module is licensed under Version 2.1 of the GNU
 * Lesser General Public License.  See lesser.txt for a text of the license.
 * CHOLMOD is also available under other licenses; contact authors for details.
 * http://www.cise.ufl.edu/research/sparse
 * -------------------------------------------------------------------------- */

/* Template routine for cholmod_solve.  Supports any numeric xtype (real,
 * complex, or zomplex).  The xtypes of all matrices (L and Y) must match.
 */

#include "cholmod_template.h"

/* ========================================================================== */
/* === simplicial template solvers ========================================== */
/* ========================================================================== */

/* LL': solve Lx=b with non-unit diagonal */
#define LL
#include "t_cholmod_lsolve.c"

/* LDL': solve LDx=b */
#define LD
#include "t_cholmod_lsolve.c"

/* LDL': solve Lx=b with unit diagonal */
#include "t_cholmod_lsolve.c"

/* LL': solve L'x=b with non-unit diagonal */
#define LL
#include "t_cholmod_ltsolve.c"

/* LDL': solve DL'x=b */
#define LD
#include "t_cholmod_ltsolve.c"

/* LDL': solve L'x=b with unit diagonal */
#include "t_cholmod_ltsolve.c"


/* ========================================================================== */
/* === t_ldl_dsolve ========================================================= */
/* ========================================================================== */

/* Solve Dx=b for an LDL' factorization, where Y holds b' on input and x' on
 * output. */

static void TEMPLATE (ldl_dsolve)
(
    cholmod_factor *L,
    cholmod_dense *Y		/* nr-by-n with leading dimension nr */
)
{
    double d [1] ;
    double *Lx, *Yx, *Yz ;
    Int *Lp ;
    Int n, nrhs, k, p, k1, k2 ;

    ASSERT (L->xtype == Y->xtype) ; /* L and Y must have the same xtype */
    ASSERT (L->n == Y->ncol) ;	    /* dimensions must match */
    ASSERT (Y->nrow == Y->d) ;	    /* leading dimension of Y = # rows of Y */
    ASSERT (L->xtype != CHOLMOD_PATTERN) ;  /* L is not symbolic */
    ASSERT (!(L->is_super) && !(L->is_ll)) ;	/* L is simplicial LDL' */

    nrhs = Y->nrow ;
    n = L->n ;
    Lp = L->p ;
    Lx = L->x ;
    Yx = Y->x ;
    Yz = Y->z ;
    for (k = 0 ; k < n ; k++)
    {
	k1 = k*nrhs ;
	k2 = (k+1)*nrhs ;
	ASSIGN_REAL (d,0, Lx,Lp[k]) ;
	for (p = k1 ; p < k2 ; p++)
	{
	    DIV_REAL (Yx,Yz,p, Yx,Yz,p, d,0) ;
	}
    }
}


/* ========================================================================== */
/* === t_simplicial_solver ================================================== */
/* ========================================================================== */

/* Solve a linear system, where Y' contains the (array-transposed) right-hand
 * side on input, and the solution on output.  No permutations are applied;
 * these must have already been applied to Y on input. */

static void TEMPLATE (simplicial_solver)
(
    int sys,		    /* system to solve */
    cholmod_factor *L,	    /* factor to use, a simplicial LL' or LDL' */
    cholmod_dense *Y	    /* right-hand-side on input, solution on output */
)
{
    if (L->is_ll)
    {
	/* The factorization is LL' */
	if (sys == CHOLMOD_A || sys == CHOLMOD_LDLt)
	{
	    /* Solve Ax=b or LL'x=b */
	    TEMPLATE (ll_lsolve_k) (L, Y) ;
	    TEMPLATE (ll_ltsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_L || sys == CHOLMOD_LD)
	{
	    /* Solve Lx=b */
	    TEMPLATE (ll_lsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_Lt || sys == CHOLMOD_DLt)
	{
	    /* Solve L'x=b */
	    TEMPLATE (ll_ltsolve_k) (L, Y) ;
	}
    }
    else
    {
	/* The factorization is LDL' */
	if (sys == CHOLMOD_A || sys == CHOLMOD_LDLt)
	{
	    /* Solve Ax=b or LDL'x=b */
	    TEMPLATE (ldl_lsolve_k) (L, Y) ;
	    TEMPLATE (ldl_dltsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_LD)
	{
	    /* Solve LDx=b */
	    TEMPLATE (ldl_ldsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_L)
	{
	    /* Solve Lx=b */
	    TEMPLATE (ldl_lsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_Lt)
	{
	    /* Solve L'x=b */
	    TEMPLATE (ldl_ltsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_DLt)
	{
	    /* Solve DL'x=b */
	    TEMPLATE (ldl_dltsolve_k) (L, Y) ;
	}
	else if (sys == CHOLMOD_D)
	{
	    /* Solve Dx=b */
	    TEMPLATE (ldl_dsolve) (L, Y) ;
	}
    }
}

#undef PATTERN
#undef REAL
#undef COMPLEX
#undef ZOMPLEX
