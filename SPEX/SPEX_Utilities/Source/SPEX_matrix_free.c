//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_matrix_free: free a SPEX_matrix
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

// Free a SPEX_matrix.  Any shallow component is not freed.
#if defined (__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include "spex_util_internal.h"

SPEX_info SPEX_matrix_free
(
    SPEX_matrix *A_handle, // matrix to free
    const SPEX_options option
)
{

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    if (!spex_initialized ( )) { return (SPEX_PANIC); } ;

    if (A_handle == NULL || (*A_handle) == NULL)
    {
        // nothing to free (not an error)
        return (SPEX_OK);
    }
    SPEX_matrix A = (*A_handle);

    //--------------------------------------------------------------------------
    // free any non-shallow components
    //--------------------------------------------------------------------------

    // free the integer pattern
    if (!(A->p_shallow)) SPEX_FREE (A->p);
    if (!(A->i_shallow)) SPEX_FREE (A->i);
    if (!(A->j_shallow)) SPEX_FREE (A->j);

    // free the values
    if (!(A->x_shallow))
    {
        switch (A->type)
        {
            case SPEX_MPZ:
                spex_free_mpz_array (&(A->x.mpz), A->nzmax) ;
                break ;

            case SPEX_MPQ:
                spex_free_mpq_array (&(A->x.mpq), A->nzmax) ;
                break ;

            case SPEX_MPFR:
                spex_free_mpfr_array (&(A->x.mpfr), A->nzmax) ;
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
    SPEX_mpq_clear (A->scale);

    //--------------------------------------------------------------------------
    // free the header
    //--------------------------------------------------------------------------

    // the header is never shallow
    SPEX_FREE (A);
    (*A_handle) = NULL ;
    return (SPEX_OK);
}

