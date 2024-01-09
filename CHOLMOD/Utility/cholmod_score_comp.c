//------------------------------------------------------------------------------
// CHOLMOD/Utility/t_cholmod_score_comp: for sorting supernodes
//------------------------------------------------------------------------------

// CHOLMOD/Utility Module. Copyright (C) 2023, Timothy A. Davis, All Rights
// Reserved.
// SPDX-License-Identifier: LGPL-2.1+

//------------------------------------------------------------------------------

#include "cholmod_internal.h"

int CHOLMOD(score_comp)
(
    struct cholmod_descendant_score_t *i,
    struct cholmod_descendant_score_t *j
)
{
    return ((i->score < j->score) ? 1 : -1) ;
}

