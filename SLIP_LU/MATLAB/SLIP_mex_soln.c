//------------------------------------------------------------------------------
// SLIP_LU/MATLAB/SLIP_mex_soln: Use SLIP LU within MATLAB
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: The .c file defining the SLIP LU MATLAB interfacee
 * This function defines: x = SLIP_mex_soln (A, b, option)
 */

#include "SLIP_LU_mex.h"

void mexFunction
(
    int nargout,
    mxArray *pargout [ ],
    int nargin,
    const mxArray *pargin [ ]
)
{
    //--------------------------------------------------------------------------
    // Initialize SLIP LU library environment
    //--------------------------------------------------------------------------

    SLIP_info status ;
    if (!slip_initialized ( ))
    {
        SLIP_MEX_OK (SLIP_initialize_expert
            (mxMalloc, mxCalloc, mxRealloc, mxFree)) ;
    }
    SuiteSparse_config.printf_func = mexPrintf ;

    //--------------------------------------------------------------------------
    // Check inputs
    //--------------------------------------------------------------------------

    if (nargout > 1 || nargin < 2 || nargin > 3)
    {
        slip_mex_error (1, "Usage: x = SLIP_mex_soln (A,b,option)") ;
    }

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (mxIsComplex (pargin [0]) || mxIsComplex (pargin [1]))
    {
        slip_mex_error (1, "inputs must be real") ;
    }
    if (!mxIsSparse (pargin [0]))     // Is the matrix sparse?
    {
        slip_mex_error (1, "first input must be sparse") ;
    }
    if (mxIsSparse (pargin [1]))         // Is b sparse?
    {
        slip_mex_error (1, "second input must be full") ;
    }

    //--------------------------------------------------------------------------
    // get the input options
    //--------------------------------------------------------------------------

    SLIP_options *option = SLIP_create_default_options();
    if (option == NULL)
    {
        slip_mex_error (SLIP_OUT_OF_MEMORY, "") ;
    }

    slip_mex_options mexoptions ;
    if (nargin > 2) slip_get_matlab_options (option, &mexoptions, pargin [2]) ;

    //--------------------------------------------------------------------------
    // get A and b
    //--------------------------------------------------------------------------

    SLIP_matrix *A = NULL ;
    SLIP_matrix *b = NULL ;
    slip_mex_get_A_and_b (&A, &b, pargin, option) ;

    if (option->print_level > 0)
    {
        printf ("\nScaled integer input matrix A:\n") ;
        SLIP_matrix_check (A, option) ;
    }

    if (option->print_level > 0)
    {
        printf ("\nScaled integer right-hand-side b:\n") ;
        SLIP_matrix_check (b, option) ;
    }

    //--------------------------------------------------------------------------
    // x = A\b via SLIP_LU, returning result as SLIP_MPQ
    //--------------------------------------------------------------------------

    SLIP_matrix *x = NULL ;
    SLIP_MEX_OK (SLIP_backslash (&x, SLIP_MPQ, A, b, option)) ;

    //--------------------------------------------------------------------------
    // print the result, if requested
    //--------------------------------------------------------------------------

    if (option->print_level > 0)
    {
        printf ("\nSolution x:\n") ;
        SLIP_matrix_check (x, option) ;
    }

    //--------------------------------------------------------------------------
    // return x to MATLAB
    //--------------------------------------------------------------------------

    if (mexoptions.solution == SLIP_SOLUTION_DOUBLE)
    {

        //----------------------------------------------------------------------
        // return x as a double MATLAB matrix
        //----------------------------------------------------------------------

        // convert x to double
        SLIP_matrix* t = NULL ;
        SLIP_MEX_OK (SLIP_matrix_copy (&t, SLIP_DENSE, SLIP_FP64, x, option)) ;
        SLIP_matrix_free (&x, NULL) ;
        x = t ;
        t = NULL ;

        // create an empty 0-by-0 MATLAB matrix and free its contents
        pargout [0] = mxCreateDoubleMatrix (0, 0, mxREAL) ;
        mxFree (mxGetDoubles (pargout [0])) ;

        // transplant x into the new MATLAB matrix and set its size
        mxSetDoubles (pargout [0], x->x.fp64) ;
        mxSetM (pargout [0], x->m) ;
        mxSetN (pargout [0], x->n) ;
        x->x.fp64 = NULL ;  // set to NULL so it is not freed by SLIP_matrix_free

    }
    else
    {

        //----------------------------------------------------------------------
        // return x as a cell array of strings
        //----------------------------------------------------------------------

        pargout [0] = mxCreateCellMatrix (x->m, x->n) ;
        int64_t mn = (x->m) * (x->n) ;
        for (int64_t p = 0 ; p < mn ; p++)
        {
            // convert x (i,j) into a C string
            char *s ;
            status = SLIP_mpfr_asprintf (&s, "%Qd", x->x.mpq [p]) ;
            if (status < 0)
            {
                slip_mex_error (1, "error converting x to string") ;
            }
            // convert the string into a MATLAB string and store in x {i,j}
            mxSetCell (pargout [0], p, mxCreateString (s)) ;
            // free the C string
            SLIP_mpfr_free_str (s) ;
        }
    }

    //--------------------------------------------------------------------------
    // free workspace
    //--------------------------------------------------------------------------

    SLIP_matrix_free (&x, option) ;
    SLIP_matrix_free (&b, option) ;
    SLIP_matrix_free (&A, option) ;
    SLIP_FREE (option) ;
    SLIP_finalize ( ) ;
}

