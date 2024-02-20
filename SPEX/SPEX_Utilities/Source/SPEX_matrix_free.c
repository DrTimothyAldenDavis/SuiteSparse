//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_matrix_free: free a SPEX_matrix
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2023, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Timothy A. Davis, and Erick Moreno-Centeno.
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

HERE
    if (!spex_initialized ( )) { return (SPEX_PANIC); } ;

HERE
    if (A_handle == NULL || (*A_handle) == NULL)
    {
        // nothing to free (not an error)
HERE
        return (SPEX_OK);
    }
    SPEX_matrix A = (*A_handle);
HERE

    //--------------------------------------------------------------------------
    // free any non-shallow components
    //--------------------------------------------------------------------------

   
        // free the integer pattern
HERE
        if (!(A->p_shallow)) SPEX_FREE (A->p);
HERE
        if (!(A->i_shallow)) SPEX_FREE (A->i);
HERE
        if (!(A->j_shallow)) SPEX_FREE (A->j);
HERE

        // free the values
        if (!(A->x_shallow))
        {
            switch (A->type)
            {
                case SPEX_MPZ:
HERE
                    if ( A->x.mpz != NULL)
                    {
HERE
                        for (int64_t i = 0; i < A->nzmax; i++)
                        {
HERE
fprintf (stderr, "i: %" PRId64" of %"PRId64"\n", i, A->nzmax) ;
fflush (stderr) ;
                            SPEX_MPZ_CLEAR( A->x.mpz[i]);
HERE
                        }
                    }
HERE
                    SPEX_FREE (A->x.mpz);
HERE
                    break ;

                case SPEX_MPQ:
HERE
                    if ( A->x.mpq != NULL)
                    {
HERE
                        for (int64_t i = 0; i < A->nzmax; i++)
                        {
HERE
                            SPEX_MPQ_CLEAR( A->x.mpq[i]);
HERE
                        }
                    }
HERE
                    SPEX_FREE (A->x.mpq);
                    break ;

                case SPEX_MPFR:
HERE
                    if ( A->x.mpfr != NULL)
                    {
HERE
                        for (int64_t i = 0; i < A->nzmax; i++)
                        {
HERE
                            SPEX_MPFR_CLEAR( A->x.mpfr[i]);
HERE
                        }
                    }
HERE
                    SPEX_FREE (A->x.mpfr);
HERE
                    break ;

                case SPEX_INT64:
HERE
                    SPEX_FREE (A->x.int64);
HERE
                    break ;

                case SPEX_FP64:
HERE
                    SPEX_FREE (A->x.fp64);
HERE
                    break ;

                default:
                    // do nothing
                    break ;
            }
        }


    // A->scale is never shallow
HERE
    SPEX_MPQ_CLEAR (A->scale);
HERE

    //--------------------------------------------------------------------------
    // free the header
    //--------------------------------------------------------------------------

    // the header is never shallow
HERE
    SPEX_FREE (A);
HERE
    (*A_handle) = NULL ;
HERE
    return (SPEX_OK);
}

