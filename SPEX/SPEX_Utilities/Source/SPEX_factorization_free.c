//------------------------------------------------------------------------------
// SPEX_Utilities/SPEX_factorization_free: Free memory for the
// SPEX_factorization data type.
//------------------------------------------------------------------------------

// SPEX_Utilities: (c) 2019-2024, Christopher Lourenco, Jinhao Chen,
// Lorena Mejia Domenzain, Erick Moreno-Centeno, and Timothy A. Davis.
// All Rights Reserved.
// SPDX-License-Identifier: GPL-2.0-or-later or LGPL-3.0-or-later

//------------------------------------------------------------------------------

/* Purpose: This function frees the memory of the SPEX_factorization struct
 *
 * Input is the SPEX_factorization structure, it is destroyed on function
 * termination.
 */

#include "spex_util_internal.h"

SPEX_info SPEX_factorization_free
(
    SPEX_factorization *F_handle,   // Structure to be deleted
    const SPEX_options option
)
{

    if (!spex_initialized ( )) return (SPEX_PANIC);

    if ((F_handle != NULL) && (*F_handle != NULL))
    {
        SPEX_mpq_clear ((*F_handle)->scale_for_A);

        SPEX_matrix_free(&((*F_handle)->L), option);
        SPEX_matrix_free(&((*F_handle)->U), option);
        SPEX_matrix_free(&((*F_handle)->rhos), option);

        SPEX_FREE((*F_handle)->P_perm);
        SPEX_FREE((*F_handle)->Pinv_perm);
        SPEX_FREE((*F_handle)->Q_perm);
        SPEX_FREE((*F_handle)->Qinv_perm);

        SPEX_FREE (*F_handle);
    }

    return (SPEX_OK);
}

