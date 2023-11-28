//------------------------------------------------------------------------------
// ParU/MATLAB/paru: MATLAB interface to ParU x=A\b
//------------------------------------------------------------------------------

// ParU Copyright (c) 2023, Mohsen Aznaveh and Timothy A. Davis, 
// All Rights Reserved.
// SPDX-License-Identifier: GPL-3.0+

//------------------------------------------------------------------------------

// Usage:
//      x = paru (A, b)
//      [x,stats] = paru (A, b, options)

#define BLAS_Intel10_64ilp
extern "C" {
#include "cholmod_matlab.h"
}
#include "ParU.hpp"

void mexFunction
(
    int	nargout,
    mxArray *pargout [ ],
    int	nargin,
    const mxArray *pargin [ ]
)
{
    double dummy = 0, rcond ;
    cholmod_sparse Amatrix, *A ;
    cholmod_dense Bmatrix, *X, *B ;
    cholmod_common Common, *cm ;
    int64_t n ;

    //--------------------------------------------------------------------------
    // start CHOLMOD
    //--------------------------------------------------------------------------

    cm = &Common ;
    cholmod_l_start (cm) ;

    /* memory management functions */
    SuiteSparse_config.malloc_func  = mxMalloc ;
    SuiteSparse_config.calloc_func  = mxCalloc ;
    SuiteSparse_config.realloc_func = mxRealloc ;
    SuiteSparse_config.free_func    = mxFree ;
    /* printf function */
    SuiteSparse_config.printf_func = mexPrintf ;
    /* math functions */
    SuiteSparse_config.hypot_func = hypot ; // was SuiteSparse_hypot in v5
    SuiteSparse_config.divcomplex_func = SuiteSparse_divcomplex ;

//  printf ("%p\n", MKL_Set_Num_Threads) ;
//  MKL_Set_Num_Threads (40) ;
    printf ("got here\n") ;

    //--------------------------------------------------------------------------
    // get inputs
    //--------------------------------------------------------------------------

    if (nargout > 2 || nargin < 2 || nargin > 3)
    {
	mexErrMsgTxt ("usage: [x,stats] = part (A,b,options)") ;
    }
    n = mxGetM (pargin [0]) ;
    if (!mxIsSparse (pargin [0]) || (n != mxGetN (pargin [0])))
    {
    	mexErrMsgTxt ("A must be square and sparse") ;
    }
    if (n != mxGetM (pargin [1]))
    {
    	mexErrMsgTxt ("# of rows of A and B must match") ;
    }
    if (mxIsSparse (pargin [1]))
    {
    	mexErrMsgTxt ("B must be dense")  ;
    }

    // get sparse matrix A
    A = sputil_get_sparse (pargin [0], &Amatrix, &dummy, 0) ;

    // get dense matrix B */
    B = sputil_get_dense (pargin [1], &Bmatrix, &dummy) ;
    int64_t nrhs = B->ncol ; 

    // create the solution X
    X = cholmod_l_allocate_dense (n, nrhs, n, CHOLMOD_REAL, cm) ;

    // get the options (TODO)

    //--------------------------------------------------------------------------
    // x = A\b using ParU
    //--------------------------------------------------------------------------

    ParU_Control Control ;
    ParU_Ret info ;
    ParU_Symbolic *Sym = NULL ;
    info = ParU_Analyze (A, &Sym, &Control) ;
    ParU_Numeric *Num = NULL ;
    info = ParU_Factorize (A, Sym, &Num, &Control) ;
    info = ParU_Solve (Sym, Num, nrhs, (double *) B->x,
        (double *) X->x, &Control) ;
    ParU_Freenum (&Num, &Control) ;
    ParU_Freesym (&Sym, &Control) ;

    //--------------------------------------------------------------------------
    // return solution to MATLAB
    //--------------------------------------------------------------------------

    pargout [0] = sputil_put_dense (&X, cm) ;
    // return statistics, if requested */
    if (nargout > 1)
    {
	// pargout [1] = ... TODO
    }

    cholmod_l_finish (cm) ;
}
