//------------------------------------------------------------------------------
// SPEX_Util/SPEX_matrix_free: free a SPEX_matrix
//------------------------------------------------------------------------------

// SPEX_Util: (c) 2019-2022, Chris Lourenco (US Naval Academy), Jinhao Chen,
// Erick Moreno-Centeno, Timothy A. Davis, Texas A&M.  All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Free a SPEX_matrix.  Any shallow component is not freed.
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "spex_util_internal.h"

SPEX_info SPEX_matrix_free
(
    SPEX_matrix **A_handle, // matrix to free
    const SPEX_options *option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!spex_initialized ( )) { return (SPEX_PANIC) ; } ;

    if (A_handle == NULL || (*A_handle) == NULL)
    {
        // nothing to free (not an error)
        return (SPEX_OK) ;
    }
    SPEX_matrix *A = (*A_handle) ;

    //--------------------------------------------------------------------------
    // free any non-shallow components
    //--------------------------------------------------------------------------

    // free the integer pattern
    if (!(A->p_shallow)) SPEX_FREE (A->p) ;
    if (!(A->i_shallow)) SPEX_FREE (A->i) ;
    if (!(A->j_shallow)) SPEX_FREE (A->j) ;

    // free the values
    if (!(A->x_shallow))
    {
        switch (A->type)
        {
            case SPEX_MPZ:
                if ( A->x.mpz)
                for (int64_t i = 0; i < A->nzmax; i++)
                {
                    if ( A->x.mpz[i] != NULL)
                    {
                        SPEX_MPZ_CLEAR( A->x.mpz[i]);
                    }
                }
                SPEX_FREE (A->x.mpz);
                break ;

            case SPEX_MPQ:
                if ( A->x.mpq)
                for (int64_t i = 0; i < A->nzmax; i++)
                {
                    if ( A->x.mpq[i] != NULL)
                    {
                        SPEX_MPQ_CLEAR( A->x.mpq[i]);
                    }
                }
                SPEX_FREE (A->x.mpq);
                break ;

            case SPEX_MPFR:
                if ( A->x.mpfr)
                for (int64_t i = 0; i < A->nzmax; i++)
                {
                    if ( A->x.mpfr[i] != NULL)
                    {
                        SPEX_MPFR_CLEAR( A->x.mpfr[i]);
                    }
                }
                SPEX_FREE (A->x.mpfr);
                break ;

            case SPEX_INT64:
                SPEX_FREE (A->x.int64) ;
                break ;

            case SPEX_FP64:
                SPEX_FREE (A->x.fp64) ;
                break ;

            default:
                // do nothing
                break ;
        }
    }

    // A->scale is never shallow
    SPEX_MPQ_CLEAR (A->scale) ;

    //--------------------------------------------------------------------------
    // free the header
    //--------------------------------------------------------------------------

    // the header is never shallow
    SPEX_FREE (A) ;
    (*A_handle) = NULL ;
    return (SPEX_OK) ;
}

