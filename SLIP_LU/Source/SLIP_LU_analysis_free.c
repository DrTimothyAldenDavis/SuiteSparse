//------------------------------------------------------------------------------
// SLIP_LU/SLIP_LU_analysis_free: Free memory from symbolic analysis struct
//------------------------------------------------------------------------------

// SLIP_LU: (c) 2019-2020, Chris Lourenco, Jinhao Chen, Erick Moreno-Centeno,
// Timothy A. Davis, Texas A&M University.  All Rights Reserved.  See
// SLIP_LU/License for the license.

//------------------------------------------------------------------------------

/* Purpose: This function frees the memory of the SLIP_LU_analysis struct
 *
 * Input is the SLIP_LU_analysis structure, it is destroyed on function
 * termination.
 */

#include "slip_internal.h"

SLIP_info SLIP_LU_analysis_free
(
    SLIP_LU_analysis **S, // Structure to be deleted
    const SLIP_options *option
)
{
    if (!slip_initialized ( )) return (SLIP_PANIC) ;

    if ((S != NULL) && (*S != NULL))
    {
        SLIP_FREE ((*S)->q) ;
        SLIP_FREE (*S) ;
    }

    return (SLIP_OK) ;
}

