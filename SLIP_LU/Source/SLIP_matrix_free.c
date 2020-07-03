//------------------------------------------------------------------------------
// SLIP_LU/SLIP_matrix_free: free a SLIP_matrix
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

// Free a SLIP_matrix.  Any shallow component is not freed.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "slip_internal.h"

SLIP_info SLIP_matrix_free
(
    SLIP_matrix **A_handle, // matrix to free
    const SLIP_options *option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!slip_initialized ( )) return (SLIP_PANIC) ;

    if (A_handle == NULL || (*A_handle) == NULL)
    {
        // nothing to free (not an error)
        return (SLIP_OK) ;
    }
    SLIP_matrix *A = (*A_handle) ;

    //--------------------------------------------------------------------------
    // free any non-shallow components
    //--------------------------------------------------------------------------

    // free the integer pattern
    if (!(A->p_shallow)) SLIP_FREE (A->p) ;
    if (!(A->i_shallow)) SLIP_FREE (A->i) ;
    if (!(A->j_shallow)) SLIP_FREE (A->j) ;

    // free the values
    if (!(A->x_shallow))
    {
        switch (A->type)
        {
            case SLIP_MPZ:
                if ( A->x.mpz)
                for (int64_t i = 0; i < A->nzmax; i++)
                {
                    if ( A->x.mpz[i] != NULL)
                    {
                        SLIP_MPZ_CLEAR( A->x.mpz[i]);
                    }
                }
                SLIP_FREE (A->x.mpz);
                break ;

            case SLIP_MPQ:
                if ( A->x.mpq)
                for (int64_t i = 0; i < A->nzmax; i++)
                {
                    if ( A->x.mpq[i] != NULL)
                    {
                        SLIP_MPQ_CLEAR( A->x.mpq[i]);
                    }
                }
                SLIP_FREE (A->x.mpq);
                break ;

            case SLIP_MPFR:
                if ( A->x.mpfr)
                for (int64_t i = 0; i < A->nzmax; i++)
                {
                    if ( A->x.mpfr[i] != NULL)
                    {
                        SLIP_MPFR_CLEAR( A->x.mpfr[i]);
                    }
                }
                SLIP_FREE (A->x.mpfr);
                break ;

            case SLIP_INT64:
                SLIP_FREE (A->x.int64) ;
                break ;

            case SLIP_FP64:
                SLIP_FREE (A->x.fp64) ;
                break ;

            default:
                // do nothing
                break ;
        }
    }

    // A->scale is never shallow
    SLIP_MPQ_CLEAR (A->scale) ;

    //--------------------------------------------------------------------------
    // free the header
    //--------------------------------------------------------------------------

    // the header is never shallow
    SLIP_FREE (A) ;
    (*A_handle) = NULL ;
    return (SLIP_OK) ;
}

